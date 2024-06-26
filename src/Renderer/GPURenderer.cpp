/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Threads/ThreadManager.h"
#include "UI/ApplicationSettings.h"

#include <Orochi/OrochiUtils.h>

GPURenderer::GPURenderer()
{
	m_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB>>();
	m_denoised_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB>>();
	m_normals_AOV_buffer = std::make_shared<OpenGLInteropBuffer<float3>>();
	m_albedo_AOV_buffer = std::make_shared<OpenGLInteropBuffer<ColorRGB>>();
}

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

	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_start_event, 0));

	launch_kernel(8, 8, resolution.x, resolution.y, launch_args);

	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, 0));
	OROCHI_CHECK_ERROR(oroEventSynchronize(m_frame_stop_event));
	OROCHI_CHECK_ERROR(oroEventElapsedTime(&m_frame_time, m_frame_start_event, m_frame_stop_event));

	m_framebuffer->unmap();
	m_normals_AOV_buffer->unmap();
	m_albedo_AOV_buffer->unmap();
}

void GPURenderer::change_render_resolution(int new_width, int new_height)
{
	m_render_width = new_width;
	m_render_height = new_height;

	m_framebuffer->resize(new_width * new_height);
	m_denoised_framebuffer->resize(new_width * new_height);
	m_normals_AOV_buffer->resize(new_width * new_height);
	m_albedo_AOV_buffer->resize(new_width * new_height);

	m_pixels_sample_count.resize(new_width * new_height);
	m_pixels_squared_luminance.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.projection_matrix = glm::transpose(glm::perspective(m_camera.vertical_fov, new_aspect, m_camera.near_plane, m_camera.far_plane));
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> GPURenderer::get_color_framebuffer()
{
	return m_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> GPURenderer::get_denoised_framebuffer()
{
	return m_denoised_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<float3>> GPURenderer::get_denoiser_normals_AOV_buffer()
{
	return m_normals_AOV_buffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> GPURenderer::get_denoiser_albedo_AOV_buffer()
{
	return m_albedo_AOV_buffer;
}

OrochiBuffer<int>& GPURenderer::get_pixels_sample_count_buffer()
{
	return m_pixels_sample_count;
}

OrochiBuffer<unsigned char>& GPURenderer::get_ray_active_buffer()
{
	return m_still_one_ray_active_buffer;
}

OrochiBuffer<unsigned int>& GPURenderer::get_stop_noise_threshold_buffer()
{
	return m_stop_noise_threshold_count_buffer;
}

HIPRTRenderSettings& GPURenderer::get_render_settings()
{
	return m_render_settings;
}

WorldSettings& GPURenderer::get_world_settings()
{
	return m_world_settings;
}

oroDeviceProp GPURenderer::get_device_properties()
{
	return m_device_properties;
}

float GPURenderer::get_gpu_frame_time()
{
	return m_frame_time;
}

float GPURenderer::get_sample_time()
{
	return get_gpu_frame_time() / ((m_render_settings.render_low_resolution ? 1 : m_render_settings.samples_per_frame));
}

HIPRTRenderData GPURenderer::get_render_data()
{
	HIPRTRenderData render_data;

	render_data.geom = m_hiprt_scene.geometry.m_geometry;

	render_data.buffers.pixels = m_framebuffer->map();
	render_data.buffers.triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.geometry.m_mesh.triangleIndices);
	render_data.buffers.vertices_positions = reinterpret_cast<float3*>(m_hiprt_scene.geometry.m_mesh.vertices);
	render_data.buffers.has_vertex_normals = reinterpret_cast<unsigned char*>(m_hiprt_scene.has_vertex_normals.get_device_pointer());
	render_data.buffers.vertex_normals = reinterpret_cast<float3*>(m_hiprt_scene.vertex_normals.get_device_pointer());
	render_data.buffers.material_indices = reinterpret_cast<int*>(m_hiprt_scene.material_indices.get_device_pointer());
	render_data.buffers.materials_buffer = reinterpret_cast<RendererMaterial*>(m_hiprt_scene.materials_buffer.get_device_pointer());
	render_data.buffers.emissive_triangles_count = m_hiprt_scene.emissive_triangles_count;
	render_data.buffers.emissive_triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_indices.get_device_pointer());

	render_data.buffers.material_textures = reinterpret_cast<oroTextureObject_t*>(m_hiprt_scene.materials_textures.get_device_pointer());
	render_data.buffers.texcoords = reinterpret_cast<float2*>(m_hiprt_scene.texcoords_buffer.get_device_pointer());
	render_data.buffers.textures_dims = reinterpret_cast<int2*>(m_hiprt_scene.textures_dims.get_device_pointer());

	// Uploading false to basically reset the flag
	unsigned char false_data = false;
	unsigned int zero_data = 0;
	m_still_one_ray_active_buffer.upload_data(&false_data);
	if (m_render_settings.stop_noise_threshold > 0.0f)
		m_stop_noise_threshold_count_buffer.upload_data(&zero_data);

	render_data.aux_buffers.denoiser_normals = m_normals_AOV_buffer->map();
	render_data.aux_buffers.denoiser_albedo = m_albedo_AOV_buffer->map();
	render_data.aux_buffers.pixel_sample_count = m_pixels_sample_count.get_device_pointer();
	render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance.get_device_pointer();
	render_data.aux_buffers.still_one_ray_active = m_still_one_ray_active_buffer.get_device_pointer();
	render_data.aux_buffers.stop_noise_threshold_count = reinterpret_cast<AtomicType<unsigned int>*>(m_stop_noise_threshold_count_buffer.get_device_pointer());

	render_data.world_settings = m_world_settings;
	render_data.render_settings = m_render_settings;

	return render_data;
}

void GPURenderer::initialize(int device_index)
{
	m_hiprt_orochi_ctx = std::make_shared<HIPRTOrochiCtx>();
	m_hiprt_orochi_ctx.get()->init(device_index);
	oroGetDeviceProperties(&m_device_properties, m_hiprt_orochi_ctx->orochi_device);

	unsigned char true_data = 1;
	m_still_one_ray_active_buffer.resize(1);
	m_still_one_ray_active_buffer.upload_data(&true_data);

	m_stop_noise_threshold_count_buffer.resize(1);

	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_start_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_stop_event));
}

void GPURenderer::compile_trace_kernel(const char* kernel_file_path, const char* kernel_function_name)
{
	std::cout << "Compiling tracer kernel \"" << kernel_function_name << "\"..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	hiprtApiFunction trace_function_out;
	std::vector<const char*> options;
	std::vector<std::string> additional_includes = { KERNEL_COMPILER_ADDITIONAL_INCLUDE, DEVICE_INCLUDES_DIRECTORY, OROCHI_INCLUDES_DIRECTORY, "-I./" };
	std::vector<std::string> macros = m_kernel_options.get_compiler_options();
	for (const std::string& macro : macros)
		options.push_back(macro.c_str());

	if (HIPPTOrochiUtils::build_trace_kernel(m_hiprt_orochi_ctx->hiprt_ctx, kernel_file_path, kernel_function_name, trace_function_out, additional_includes, options, 0, 1, false) != hiprtError::hiprtSuccess)
	{
		std::cerr << "Unable to compile kernel \"" << kernel_function_name << "\". Cannot continue." << std::endl;
		int ignored = std::getchar();
		std::exit(1);
	}

	std::cout << std::endl;

	m_trace_kernel = *reinterpret_cast<oroFunction*>(&trace_function_out);

	int numRegs = 0;
	int numSmem = 0;

	if ((m_device_properties.major >= 7 && m_device_properties.minor >= 5) || std::string(m_device_properties.name).find("Radeon") != std::string::npos)
	{
		// Getting the number of registers / shared memory of the function
		// doesn't work on a CC 5.2 NVIDIA GPU but does work on a 7.5 so
		// I'm assuming this is an issue of compute capability (although
		// it could be a completely different thing)
		//
		// Also this if() statement assumes that getting the number of
		// registers will work on any AMD card but this has only been
		// tested on a gfx1100 GPU. This should be tested on older hardware
		// if possible
		OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, m_trace_kernel));
		OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, m_trace_kernel));
	}

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Trace kernel: " << numRegs << " registers, shared memory " << numSmem << std::endl;
	std::cout << "Kernel \"" << kernel_function_name << "\" compiled in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void GPURenderer::set_kernel_option(const std::string& name, int value)
{
	m_kernel_options.set_option(name, value);
}

int GPURenderer::get_kernel_option_value(const std::string& name)
{
	return m_kernel_options.get_option_value(name);
}

int* GPURenderer::get_kernel_option_pointer(const std::string& name)
{
	return m_kernel_options.get_pointer_to_option_value(name);
}

void GPURenderer::launch_kernel(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args)
{
	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(static_cast<float>(res_x) / tile_size_x);
	nb_groups.y = std::ceil(static_cast<float>(res_y) / tile_size_y);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_trace_kernel, nb_groups.x, nb_groups.y, 1, tile_size_x, tile_size_y, 1, 0, 0, launch_args, 0));
}

void GPURenderer::set_hiprt_scene_from_scene(const Scene& scene)
{
	HIPRTScene& hiprt_scene = m_hiprt_scene;
	HIPRTGeometry& geometry = hiprt_scene.geometry;
	
	geometry.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	geometry.upload_indices(scene.triangle_indices);
	geometry.upload_vertices(scene.vertices_positions);
	geometry.build_bvh();

	hiprt_scene.has_vertex_normals.resize(scene.has_vertex_normals.size());
	hiprt_scene.has_vertex_normals.upload_data(scene.has_vertex_normals.data());

	hiprt_scene.vertex_normals.resize(scene.vertex_normals.size());
	hiprt_scene.vertex_normals.upload_data(scene.vertex_normals.data());

	hiprt_scene.material_indices.resize(scene.material_indices.size());
	hiprt_scene.material_indices.upload_data(scene.material_indices.data());

	hiprt_scene.materials_buffer.resize(scene.materials.size());
	hiprt_scene.materials_buffer.upload_data(scene.materials.data());

	hiprt_scene.emissive_triangles_count = scene.emissive_triangle_indices.size();
	if (hiprt_scene.emissive_triangles_count > 0)
	{
		hiprt_scene.emissive_triangles_indices.resize(scene.emissive_triangle_indices.size());
		hiprt_scene.emissive_triangles_indices.upload_data(scene.emissive_triangle_indices.data());
	}

	hiprt_scene.texcoords_buffer.resize(scene.texcoords.size());
	hiprt_scene.texcoords_buffer.upload_data(scene.texcoords.data());

	// We're joining the threads that were loading the scene textures in the background
	// at the last moment so that they had the maximum amount of time to load the textures
	// while the main thread was doing something else
	ThreadManager::join_threads(ThreadManager::TEXTURE_THREADS_KEY);

	if (scene.textures.size() > 0)
	{
		std::vector<oroTextureObject_t> oro_textures(scene.textures.size());
		m_materials_textures.reserve(scene.textures.size());
		for (int i = 0; i < scene.textures.size(); i++)
		{
			// We need to keep the texture alive so they are not destroyed when returning from 
			// this function so we're adding them to a member buffer
			m_materials_textures.push_back(OrochiTexture(scene.textures[i]));

			oro_textures[i] = m_materials_textures.back().get_device_texture();
		}

		hiprt_scene.materials_textures.resize(oro_textures.size());
		hiprt_scene.materials_textures.upload_data(oro_textures.data());

		hiprt_scene.textures_dims.resize(scene.textures_dims.size());
		hiprt_scene.textures_dims.upload_data(scene.textures_dims.data());
	}
}

void GPURenderer::set_scene(const Scene& scene)
{
	set_hiprt_scene_from_scene(scene);

	m_materials = scene.materials;
	m_material_names = scene.material_names;
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

const std::vector<std::string>& GPURenderer::get_material_names()
{
	return m_material_names;
}

void GPURenderer::update_materials(std::vector<RendererMaterial>& materials)
{
	m_materials = materials;
	m_hiprt_scene.materials_buffer.upload_data(materials.data());
}

void GPURenderer::set_camera(const Camera& camera)
{
	m_camera = camera;
}

void GPURenderer::toggle_adaptive_sampling_buffers(bool adaptive_sampling_enabled)
{
	if (adaptive_sampling_enabled)
	{
		m_pixels_squared_luminance.resize(m_render_width * m_render_height);
		m_pixels_sample_count.resize(m_render_width * m_render_height);
	}
	else
	{
		m_pixels_squared_luminance.free();
		m_pixels_sample_count.free();
	}
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
