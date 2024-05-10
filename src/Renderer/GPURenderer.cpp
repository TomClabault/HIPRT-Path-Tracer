/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include <algorithm>
#include <deque>

#include <Orochi/OrochiUtils.h>
#include "Renderer/GPURenderer.h"

void GPURenderer::render()
{
	int tile_size_x = 8;
	int tile_size_y = 8;

	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(m_render_width / (float)tile_size_x);
	nb_groups.y = std::ceil(m_render_height / (float)tile_size_y);

	hiprtInt2 resolution = make_hiprtInt2(m_render_width, m_render_height);

	HIPRTCamera hiprt_cam = m_camera.to_hiprt();
	HIPRTRenderData render_data = get_render_data();
	void* launch_args[] = { &render_data, &resolution, &hiprt_cam};
	launch_kernel(8, 8, resolution.x, resolution.y, launch_args);

#ifndef OROCHI_ENABLE_CUEW
	// We only want to unmap for OpenGL interop buffers that are only available
	// on AMD (for now)
	m_pixels_interop_buffer.unmap();
#endif
}

void GPURenderer::change_render_resolution(int new_width, int new_height)
{
	m_render_width = new_width;
	m_render_height = new_height;

	m_pixels_interop_buffer.resize(new_width * new_height);
	m_normals_buffer.resize(new_width * new_height);
	m_albedo_buffer.resize(new_width * new_height);

	m_pixels_sample_count.resize(new_width * new_height);
	m_pixels_squared_luminance.resize(new_width * new_height);
	m_debug_pixel_active.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.projection_matrix = glm::transpose(glm::perspective(m_camera.vertical_fov, new_aspect, m_camera.near_plane, m_camera.far_plane));
}

InteropBufferType<ColorRGB>& GPURenderer::get_color_framebuffer()
{
	return m_pixels_interop_buffer;
}

OrochiBuffer<ColorRGB>& GPURenderer::get_denoiser_albedo_buffer()
{
	return m_albedo_buffer;
}

OrochiBuffer<float3>& GPURenderer::get_denoiser_normals_buffer()
{
	return m_normals_buffer;
}

OrochiBuffer<int>& GPURenderer::get_debug_pixel_active_buffer()
{
	return m_debug_pixel_active;
}

OrochiBuffer<int>& GPURenderer::get_pixels_sample_count_buffer()
{
	return m_pixels_sample_count;
}

HIPRTRenderSettings& GPURenderer::get_render_settings()
{
	return m_render_settings;
}

WorldSettings& GPURenderer::get_world_settings()
{
	return m_world_settings;
}

int GPURenderer::get_sample_number()
{
	return m_render_settings.sample_number;
}

void GPURenderer::set_sample_number(int sample_number)
{
	m_render_settings.sample_number = sample_number;
}

HIPRTRenderData GPURenderer::get_render_data()
{
	HIPRTRenderData render_data;

	render_data.geom = m_hiprt_scene.geometry;

	render_data.buffers.pixels = m_pixels_interop_buffer.get_device_pointer();
	render_data.buffers.triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.mesh.triangleIndices);
	render_data.buffers.triangles_vertices = reinterpret_cast<float3*>(m_hiprt_scene.mesh.vertices);
	render_data.buffers.has_vertex_normals = reinterpret_cast<unsigned char*>(m_hiprt_scene.has_vertex_normals);
	render_data.buffers.vertex_normals = reinterpret_cast<float3*>(m_hiprt_scene.vertex_normals);
	render_data.buffers.material_indices = reinterpret_cast<int*>(m_hiprt_scene.material_indices);
	render_data.buffers.materials_buffer = reinterpret_cast<RendererMaterial*>(m_hiprt_scene.materials_buffer);
	render_data.buffers.emissive_triangles_count = m_hiprt_scene.emissive_triangles_count;
	render_data.buffers.emissive_triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_indices);
	render_data.buffers.material_textures = reinterpret_cast<oroTextureObject_t*>(m_hiprt_scene.material_textures);
	render_data.buffers.texcoords = reinterpret_cast<float2*>(m_hiprt_scene.texcoords_buffer);

	render_data.aux_buffers.debug_pixel_active = m_debug_pixel_active.get_device_pointer();
	render_data.aux_buffers.denoiser_normals = m_normals_buffer.get_device_pointer();
	render_data.aux_buffers.denoiser_albedo = m_albedo_buffer.get_device_pointer();
	render_data.aux_buffers.pixel_sample_count = m_pixels_sample_count.get_device_pointer();
	render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance.get_device_pointer();

	render_data.world_settings = m_world_settings;
	render_data.render_settings = m_render_settings;

	return render_data;
}

void GPURenderer::init_ctx(int device_index)
{
	m_hiprt_orochi_ctx = std::make_shared<HIPRTOrochiCtx>();
	m_hiprt_orochi_ctx.get()->init(device_index);
}

void GPURenderer::compile_trace_kernel(const char* kernel_file_path, const char* kernel_function_name)
{
	std::cout << "Compiling tracer kernel \"" << kernel_function_name << "\"..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::pair<std::string, std::string>> precompiler_defines;
	std::vector<const char*> options;

	std::vector<std::string> additional_includes = { KERNEL_COMPILER_ADDITIONAL_INCLUDE, DEVICE_INCLUDES_DIRECTORY, OROCHI_INCLUDES_DIRECTORY, "-I./" };

	hiprtApiFunction trace_function_out;
	if (HIPPTOrochiUtils::build_trace_kernel(m_hiprt_orochi_ctx->hiprt_ctx, kernel_file_path, kernel_function_name, trace_function_out, additional_includes, options, 0, 1, false) != hiprtError::hiprtSuccess)
	{
		std::cerr << "Unable to compile kernel \"" << kernel_function_name << "\". Cannot continue." << std::endl;
		std::getchar();
		std::exit(1);
	}

	m_trace_kernel = *reinterpret_cast<oroFunction*>(&trace_function_out);

	int numRegs;
	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, m_trace_kernel));

	int numSmem;
	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, m_trace_kernel));

	std::cout << "Trace kernel: registers " << numRegs << ", shared memory " << numSmem << std::endl;
	
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Kernel \"" << kernel_function_name << "\" compiled in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void GPURenderer::launch_kernel(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args)
{
	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(static_cast<float>(res_x) / tile_size_x);
	nb_groups.y = std::ceil(static_cast<float>(res_y) / tile_size_y);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_trace_kernel, nb_groups.x, nb_groups.y, 1, tile_size_x, tile_size_y, 1, 0, 0, launch_args, 0));
}

// TODO write a logger
void log_bvh_building(hiprtBuildFlags build_flags)
{
	std::cout << "Compiling BVH building kernels & building scene ";
	if (build_flags  == 0)
		// This is hiprtBuildFlagBitPreferFastBuild
		std::cout << "LBVH";
	else if (build_flags & hiprtBuildFlagBitPreferBalancedBuild)
		std::cout << "PLOC BVH";
	else if (build_flags & hiprtBuildFlagBitPreferHighQualityBuild)
		std::cout << "SBVH";

	std::cout << "... (This can take 30s+ on NVIDIA hardware)" << std::endl;
}

void GPURenderer::set_hiprt_scene_from_scene(Scene& scene)
{
	m_hiprt_scene = HIPRTScene(m_hiprt_orochi_ctx->hiprt_ctx);
	HIPRTScene& hiprt_scene = m_hiprt_scene;
	
	hiprtTriangleMeshPrimitive& mesh = hiprt_scene.mesh;

	// Allocating and initializing the indices buffer
	mesh.triangleCount = scene.triangle_indices.size() / 3;
	mesh.triangleStride = sizeof(int3);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&mesh.triangleIndices), mesh.triangleCount * sizeof(int3)));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(mesh.triangleIndices), scene.triangle_indices.data(), mesh.triangleCount * sizeof(int3)));

	// Allocating and initializing the vertices positions buiffer
	mesh.vertexCount = scene.vertices_positions.size();
	mesh.vertexStride = sizeof(float3);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&mesh.vertices), mesh.vertexCount * sizeof(float3)));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(mesh.vertices), scene.vertices_positions.data(), mesh.vertexCount * sizeof(float3)));

	hiprtGeometryBuildInput geometry_build_input;
	geometry_build_input.type = hiprtPrimitiveTypeTriangleMesh;
	geometry_build_input.primitive.triangleMesh = hiprt_scene.mesh;

	// Getting the buffer sizes for the construction of the BVH
	size_t geometry_temp_size;
	hiprtDevicePtr geometry_temp;
	hiprtBuildOptions build_options;
	build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	HIPRT_CHECK_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(m_hiprt_orochi_ctx->hiprt_ctx, geometry_build_input, build_options, geometry_temp_size));
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&geometry_temp), geometry_temp_size));

	// Building the BVH
	log_bvh_building(build_options.buildFlags);
	hiprtGeometry& scene_geometry = hiprt_scene.geometry;
	HIPRT_CHECK_ERROR(hiprtCreateGeometry(m_hiprt_orochi_ctx->hiprt_ctx, geometry_build_input, build_options, scene_geometry));
	HIPRT_CHECK_ERROR(hiprtBuildGeometry(m_hiprt_orochi_ctx->hiprt_ctx, hiprtBuildOperationBuild, geometry_build_input, build_options, geometry_temp, 0, scene_geometry));

	OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(geometry_temp)));

	hiprtDevicePtr normals_present_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&normals_present_buffer), sizeof(unsigned char) * scene.has_vertex_normals.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(normals_present_buffer), scene.has_vertex_normals.data(), sizeof(unsigned char) * scene.has_vertex_normals.size()));
	hiprt_scene.has_vertex_normals = normals_present_buffer;

	hiprtDevicePtr vertex_normals_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&vertex_normals_buffer), sizeof(float3) * scene.vertex_normals.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(vertex_normals_buffer), scene.vertex_normals.data(), sizeof(float3) * scene.vertex_normals.size()));
	hiprt_scene.vertex_normals = vertex_normals_buffer;

	hiprtDevicePtr material_indices_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&material_indices_buffer), sizeof(int) * scene.material_indices.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(material_indices_buffer), scene.material_indices.data(), sizeof(int) * scene.material_indices.size()));
	hiprt_scene.material_indices = material_indices_buffer;

	hiprtDevicePtr materials_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&materials_buffer), sizeof(RendererMaterial) * scene.materials.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(materials_buffer), scene.materials.data(), sizeof(RendererMaterial) * scene.materials.size()));
	hiprt_scene.materials_buffer = materials_buffer;

	hiprt_scene.emissive_triangles_count = scene.emissive_triangle_indices.size();

	hiprtDevicePtr emissive_triangle_indices;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&emissive_triangle_indices), sizeof(int) * scene.emissive_triangle_indices.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(emissive_triangle_indices), scene.emissive_triangle_indices.data(), sizeof(int) * scene.emissive_triangle_indices.size()));
	hiprt_scene.emissive_triangles_indices = emissive_triangle_indices;

	std::vector<oroTextureObject_t> oro_textures;
	oro_textures.reserve(scene.textures.size());
	m_materials_textures.reserve(scene.textures.size());
	for (const ImageRGBA& texture : scene.textures)
	{
		// We need to keep the texture alive so they are not destroyed when returning from 
		// this function so we're adding them to a member buffer
		m_materials_textures.push_back(OrochiTexture(texture));
		oro_textures.push_back(m_materials_textures.back().get_device_texture());
	}

	hiprtDevicePtr material_textures;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&material_textures), sizeof(oroTextureObject_t) * scene.textures.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(material_textures), oro_textures.data(), sizeof(oroTextureObject_t) * oro_textures.size()));
	hiprt_scene.material_textures = material_textures;

	hiprtDevicePtr texcoords_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&texcoords_buffer), sizeof(float2) * scene.texcoords.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(texcoords_buffer), scene.texcoords.data(), sizeof(float2) * scene.texcoords.size()));
	hiprt_scene.texcoords_buffer = texcoords_buffer;
}

void GPURenderer::set_scene(Scene& scene)
{
	set_hiprt_scene_from_scene(scene);

	m_materials = scene.materials;
}

void GPURenderer::set_envmap(ImageRGBA& envmap_image)
{
	m_envmap.init_from_image(envmap_image);
	m_envmap.compute_cdf(envmap_image);

	m_world_settings.envmap = m_envmap.get_device_texture();
	m_world_settings.envmap_width = m_envmap.width;
	m_world_settings.envmap_height = m_envmap.height;
	m_world_settings.envmap_cdf = m_envmap.get_cdf_device_pointer();
}

const std::vector<RendererMaterial>& GPURenderer::get_materials()
{
	return m_materials;
}

void GPURenderer::update_materials(std::vector<RendererMaterial>& materials)
{
	m_materials = materials;

	if (m_hiprt_scene.materials_buffer)
		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_hiprt_scene.materials_buffer)));

	hiprtDevicePtr materials_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&materials_buffer), sizeof(RendererMaterial) * materials.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(materials_buffer), materials.data(), sizeof(RendererMaterial) * materials.size()));
	m_hiprt_scene.materials_buffer = materials_buffer;
}

void GPURenderer::set_camera(const Camera& camera)
{
	m_camera = camera;
}

void GPURenderer::translate_camera_view(glm::vec3 translation)
{
	m_camera.translation = m_camera.translation + translation * glm::conjugate(m_camera.rotation);
}

void GPURenderer::rotate_camera_view(glm::vec3 rotation_angles)
{
	glm::quat qx = glm::angleAxis(rotation_angles.y, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::quat qy = glm::angleAxis(rotation_angles.x, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::quat orientation = glm::normalize(qy * m_camera.rotation * qx);
	m_camera.rotation = orientation;
}

void GPURenderer::zoom_camera_view(float offset)
{
	glm::vec3 translation(0, 0, offset);
	m_camera.translation = m_camera.translation + translation * glm::conjugate(m_camera.rotation);
}
