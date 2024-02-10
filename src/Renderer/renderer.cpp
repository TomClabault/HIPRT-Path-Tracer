#include "renderer.h"

void Renderer::render()
{
	int tile_size_x = 8;
	int tile_size_y = 8;

	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(m_framebuffer_width / (float)tile_size_x);
	nb_groups.y = std::ceil(m_framebuffer_height / (float)tile_size_y);

	hiprtInt2 resolution = make_hiprtInt2(m_framebuffer_width, m_framebuffer_height);

	HIPRTCamera hiprt_cam = m_camera.to_hiprt();
	void* launch_args[] = { &m_scene.geometry, &m_scene_data, m_framebuffer.get_pointer_address(), &resolution, &hiprt_cam};
	launch_kernel(8, 8, resolution.x, resolution.y, launch_args);
}

void Renderer::resize_frame(int new_width, int new_height)
{
	m_framebuffer_width = new_width;
	m_framebuffer_height = new_height;

	// * 4 for RGBA
	m_framebuffer.resize(new_width * new_height * 4);
}

OrochiBuffer<float>& Renderer::get_orochi_framebuffer()
{
	return m_framebuffer;
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

Renderer::HIPRTScene Renderer::create_hiprt_scene_from_scene(Scene& scene)
{
	Renderer::HIPRTScene hiprt_scene(m_hiprt_orochi_ctx->hiprt_ctx);
	hiprtTriangleMeshPrimitive& mesh = hiprt_scene.mesh;

	// Allocating and initializing the indices buffer
	mesh.triangleCount = scene.vertices_indices.size() / 3;
	mesh.triangleStride = sizeof(hiprtInt3);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&mesh.triangleIndices), mesh.triangleCount * sizeof(hiprtInt3)));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(mesh.triangleIndices), scene.vertices_indices.data(), mesh.triangleCount * sizeof(hiprtInt3)));

	// Allocating and initializing the vertices positions buiffer
	mesh.vertexCount = scene.vertices_positions.size();
	mesh.vertexStride = sizeof(hiprtFloat3);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&mesh.vertices), mesh.vertexCount * sizeof(hiprtFloat3)));
	OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(mesh.vertices), scene.vertices_positions.data(), mesh.vertexCount * sizeof(hiprtFloat3)));

	hiprtGeometryBuildInput geometry_build_input;
	geometry_build_input.type = hiprtPrimitiveTypeTriangleMesh;
	geometry_build_input.primitive.triangleMesh = hiprt_scene.mesh;

	// Getting the buffer sizes for the construction of the BVH
	size_t geometry_temp_size;
	hiprtDevicePtr geometry_temp;
	hiprtBuildOptions build_options;
	build_options.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;// TODO ImGui to choose the flags at runtime and be able to compare the performance

	HIPRT_CHECK_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(m_hiprt_orochi_ctx->hiprt_ctx, geometry_build_input, build_options, geometry_temp_size));
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&geometry_temp), geometry_temp_size));

	// Building the BVH
	hiprtGeometry& scene_geometry = hiprt_scene.geometry;
	HIPRT_CHECK_ERROR(hiprtCreateGeometry(m_hiprt_orochi_ctx->hiprt_ctx, geometry_build_input, build_options, scene_geometry));
	HIPRT_CHECK_ERROR(hiprtBuildGeometry(m_hiprt_orochi_ctx->hiprt_ctx, hiprtBuildOperationBuild, geometry_build_input, build_options, geometry_temp, 0, scene_geometry));

	OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(geometry_temp)));

	return hiprt_scene;
}

void Renderer::set_hiprt_scene(const Renderer::HIPRTScene& scene)
{
	m_scene = scene;
	m_scene_data.triangles_indices = reinterpret_cast<int*>(scene.mesh.triangleIndices);
	m_scene_data.triangles_vertices = reinterpret_cast<hiprtFloat3*>(scene.mesh.vertices);
}

void Renderer::set_camera(const Camera& camera)
{
	m_camera = camera;
}

void Renderer::translate_camera_view(glm::vec3 translation)
{
	m_camera.translation = m_camera.translation + translation * glm::conjugate(m_camera.rotation);

	/*glm::mat4x4 view_mat = m_camera.get_view_matrix();

	m_camera.translation = m_camera.translation + glm::vec3(view_mat[0][0], view_mat[1][0], view_mat[2][0]) * translation.x;
	m_camera.translation = m_camera.translation + glm::vec3(view_mat[0][1], view_mat[1][1], view_mat[2][1]) * translation.y;*/
}

void Renderer::rotate_camera_view(glm::vec3 rotation_angles)
{
	glm::quat qx = glm::angleAxis(rotation_angles.y, glm::vec3(1.0f, 0.0f, 0.0f));
	glm::quat qy = glm::angleAxis(rotation_angles.x, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::quat orientation = glm::normalize(qy * m_camera.rotation * qx);
	m_camera.rotation = orientation;

	/*glm::quat rotation_x = glm::quat(glm::vec3(rotation_angles.y, 0, 0));
	glm::quat rotation_y = glm::quat(glm::vec3(0, -rotation_angles.x, 0));

	rotation_x = glm::normalize(rotation_x);
	rotation_y = glm::normalize(rotation_y);

	m_camera.rotation = rotation_x * m_camera.rotation * rotation.y*/;
}

void Renderer::zoom_camera_view(float offset)
{
	glm::vec3 translation(0, 0, offset);
	m_camera.translation = m_camera.translation + translation * glm::conjugate(m_camera.rotation);

	//m_camera.translation = m_camera.translation + m_camera.get_view_direction() * offset;
}
