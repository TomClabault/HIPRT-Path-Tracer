#include "renderer.h"

void Renderer::render()
{
	int tile_size_x = 8;
	int tile_size_y = 8;

	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(m_render_width / (float)tile_size_x);
	nb_groups.y = std::ceil(m_render_height / (float)tile_size_y);

	hiprtInt2 resolution = make_hiprtInt2(m_render_width, m_render_height);

	HIPRTCamera hiprt_cam = m_camera.to_hiprt();
	HIPRTRenderData render_data = get_render_data();
	void* launch_args[] = { &m_scene.get()->geometry, &render_data, &resolution, &hiprt_cam};
	launch_kernel(8, 8, resolution.x, resolution.y, launch_args);
}

void Renderer::change_render_resolution(int new_width, int new_height)
{
	m_render_width = new_width;
	m_render_height = new_height;

	m_pixels_buffer.resize(new_width * new_height * 3);
	m_ws_normals_buffer.resize(new_width * new_height);
	m_albedo_buffer.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.projection_matrix = glm::transpose(glm::perspective(m_camera.vertical_fov, new_aspect, m_camera.near_plane, m_camera.far_plane));
}

OrochiBuffer<float>& Renderer::get_orochi_framebuffer()
{
	return m_pixels_buffer;
}

RenderSettings& Renderer::get_render_settings()
{
	return m_render_settings;
}

void Renderer::set_sample_number(int sample_number)
{
	m_render_settings.sample_number = sample_number;
}

HIPRTRenderData Renderer::get_render_data()
{
	HIPRTRenderData render_data;

	render_data.geom = m_scene.get()->geometry;
	render_data.pixels = m_pixels_buffer.get_pointer();
	render_data.ws_normals = m_ws_normals_buffer.get_pointer();
	render_data.albedo = m_albedo_buffer.get_pointer();
	render_data.triangles_indices = reinterpret_cast<int*>(m_scene.get()->mesh.triangleIndices);
	render_data.triangles_vertices = reinterpret_cast<hiprtFloat3*>(m_scene.get()->mesh.vertices);
	render_data.normals_present = reinterpret_cast<unsigned char*>(m_scene.get()->normals_present);
	render_data.vertex_normals = reinterpret_cast<hiprtFloat3*>(m_scene.get()->vertex_normals);
	render_data.material_indices = reinterpret_cast<int*>(m_scene.get()->material_indices);
	render_data.materials_buffer = reinterpret_cast<RendererMaterial*>(m_scene.get()->materials_buffer);
	render_data.emissive_triangles_count = m_scene.get()->emissive_triangles_count;
	render_data.emissive_triangles_indices = reinterpret_cast<int*>(m_scene.get()->emissive_triangles_indices);

	render_data.render_settings.sample_number = m_render_settings.sample_number;
	render_data.render_settings.nb_bounces = m_render_settings.nb_bounces;
	render_data.render_settings.samples_per_frame = m_render_settings.samples_per_frame;

	return render_data;
}

void Renderer::init_ctx(int device_index)
{
	m_hiprt_orochi_ctx = std::make_shared<Renderer::HIPRTOrochiCtx>();
	m_hiprt_orochi_ctx.get()->init(device_index);
}

void Renderer::compile_trace_kernel(const char* kernel_file_path, const char* kernel_function_name)
{
	std::vector<std::string> include_paths{"./", "../thirdparties/hiprt/include"};
	std::vector<std::pair<std::string, std::string>> precompiler_defines;
	buildTraceKernelFromBitcode(m_hiprt_orochi_ctx->hiprt_ctx, kernel_file_path, kernel_function_name, m_trace_kernel, include_paths);
}

void Renderer::launch_kernel(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args)
{
	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(static_cast<float>(res_x) / tile_size_x);
	nb_groups.y = std::ceil(static_cast<float>(res_y) / tile_size_y);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_trace_kernel, nb_groups.x, nb_groups.y, 1, tile_size_x, tile_size_y, 1, 0, 0, launch_args, 0));
}

std::shared_ptr<Renderer::HIPRTScene> Renderer::create_hiprt_scene_from_scene(Scene& scene)
{
	std::shared_ptr<Renderer::HIPRTScene> hiprt_scene_ptr = std::make_shared<Renderer::HIPRTScene>(m_hiprt_orochi_ctx->hiprt_ctx); 
	Renderer::HIPRTScene* const hiprt_scene = hiprt_scene_ptr.get();
	
	hiprtTriangleMeshPrimitive& mesh = hiprt_scene->mesh;

	// Allocating and initializing the indices buffer
	mesh.triangleCount = scene.triangle_indices.size() / 3;
	mesh.triangleStride = sizeof(hiprtInt3);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&mesh.triangleIndices), mesh.triangleCount * sizeof(hiprtInt3)));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(mesh.triangleIndices), scene.triangle_indices.data(), mesh.triangleCount * sizeof(hiprtInt3)));

	// Allocating and initializing the vertices positions buiffer
	mesh.vertexCount = scene.vertices_positions.size();
	mesh.vertexStride = sizeof(hiprtFloat3);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&mesh.vertices), mesh.vertexCount * sizeof(hiprtFloat3)));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(mesh.vertices), scene.vertices_positions.data(), mesh.vertexCount * sizeof(hiprtFloat3)));

	hiprtGeometryBuildInput geometry_build_input;
	geometry_build_input.type = hiprtPrimitiveTypeTriangleMesh;
	geometry_build_input.primitive.triangleMesh = hiprt_scene->mesh;

	// Getting the buffer sizes for the construction of the BVH
	size_t geometry_temp_size;
	hiprtDevicePtr geometry_temp;
	hiprtBuildOptions build_options;
	build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	HIPRT_CHECK_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(m_hiprt_orochi_ctx->hiprt_ctx, geometry_build_input, build_options, geometry_temp_size));
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&geometry_temp), geometry_temp_size));

	// Building the BVH
	hiprtGeometry& scene_geometry = hiprt_scene->geometry;
	HIPRT_CHECK_ERROR(hiprtCreateGeometry(m_hiprt_orochi_ctx->hiprt_ctx, geometry_build_input, build_options, scene_geometry));
	HIPRT_CHECK_ERROR(hiprtBuildGeometry(m_hiprt_orochi_ctx->hiprt_ctx, hiprtBuildOperationBuild, geometry_build_input, build_options, geometry_temp, 0, scene_geometry));

	OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(geometry_temp)));

	hiprtDevicePtr normals_present_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&normals_present_buffer), sizeof(unsigned char) * scene.normals_present.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(normals_present_buffer), scene.normals_present.data(), sizeof(unsigned char) * scene.normals_present.size()));
	hiprt_scene_ptr.get()->normals_present = normals_present_buffer;

	hiprtDevicePtr vertex_normals_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&vertex_normals_buffer), sizeof(hiprtFloat3) * scene.vertex_normals.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(vertex_normals_buffer), scene.vertex_normals.data(), sizeof(hiprtFloat3) * scene.vertex_normals.size()));
	hiprt_scene_ptr.get()->vertex_normals = vertex_normals_buffer;

	hiprtDevicePtr material_indices_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&material_indices_buffer), sizeof(int) * scene.material_indices.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(material_indices_buffer), scene.material_indices.data(), sizeof(int) * scene.material_indices.size()));
	hiprt_scene_ptr.get()->material_indices = material_indices_buffer;

	hiprtDevicePtr materials_buffer;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&materials_buffer), sizeof(RendererMaterial) * scene.materials.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(materials_buffer), scene.materials.data(), sizeof(RendererMaterial) * scene.materials.size()));
	hiprt_scene_ptr.get()->materials_buffer = materials_buffer;

	hiprt_scene_ptr.get()->emissive_triangles_count = scene.emissive_triangle_indices.size();

	hiprtDevicePtr emissive_triangle_indices;
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&emissive_triangle_indices), sizeof(int) * scene.emissive_triangle_indices.size()));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(emissive_triangle_indices), scene.emissive_triangle_indices.data(), sizeof(int) * scene.emissive_triangle_indices.size()));
	hiprt_scene_ptr.get()->emissive_triangles_indices = emissive_triangle_indices;

	return hiprt_scene_ptr;
}

void Renderer::set_hiprt_scene(std::shared_ptr<Renderer::HIPRTScene> scene)
{
	m_scene = scene;
}

void Renderer::set_camera(const Camera& camera)
{
	m_camera = camera;
}

void Renderer::translate_camera_view(glm::vec3 translation)
{
	m_camera.translation = m_camera.translation + translation * glm::conjugate(m_camera.rotation);
}

void Renderer::rotate_camera_view(glm::vec3 rotation_angles)
{
	glm::quat qx = glm::angleAxis(rotation_angles.y, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::quat qy = glm::angleAxis(rotation_angles.x, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::quat orientation = glm::normalize(qy * m_camera.rotation * qx);
	m_camera.rotation = orientation;
}

void Renderer::zoom_camera_view(float offset)
{
	glm::vec3 translation(0, 0, offset);
	m_camera.translation = m_camera.translation + translation * glm::conjugate(m_camera.rotation);
}
