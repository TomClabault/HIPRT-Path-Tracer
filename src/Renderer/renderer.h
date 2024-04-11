#ifndef RENDERER_H
#define RENDERER_H

#include "glm/gtc/matrix_transform.hpp"
#include "HIPRT-Orochi/orochi_buffer.h"
#include "HIPRT-Orochi/hiprt_orochi_ctx.h"
#include "HIPRT-Orochi/hiprt_scene.h"
#include "Kernels/includes/hiprt_render_data.h"
#include "Renderer/render_settings.h"
#include "Scene/camera.h"
#include "Scene/scene_parser.h"

#include "open_image_denoiser.h"

#include <vector>

class Renderer
{
public:
	Renderer(int width, int height, HIPRTOrochiCtx* hiprt_orochi_ctx) : 
		m_render_width(width), m_render_height(height),
		m_pixels_buffer(width * height * 3), m_normals_buffer(width * height),  // TODO buffer initialization necessary here ?
		m_albedo_buffer(width * height), m_hiprt_orochi_ctx(hiprt_orochi_ctx),
		m_trace_kernel(nullptr)
	{
		m_hiprt_scene.hiprt_ctx = hiprt_orochi_ctx->hiprt_ctx;
	}

	Renderer() : m_hiprt_scene(nullptr) {}

	void render(const OpenImageDenoiser& denoiser);
	void change_render_resolution(int new_width, int new_height);

	OrochiBuffer<Color>& get_color_framebuffer();
	OrochiBuffer<Color>& get_denoiser_albedo_buffer();
	OrochiBuffer<hiprtFloat3>& get_denoiser_normals_buffer();
	RenderSettings& get_render_settings();
	HIPRTRenderData get_render_data(const OpenImageDenoiser& denoiser);

	void init_ctx(int device_index);
	void compile_trace_kernel(const char* kernel_file_path, const char* kernel_function_name);
	void launch_kernel(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args);

	void set_scene(Scene& scene);
	const std::vector<RendererMaterial>& get_materials();
	void update_materials(std::vector<RendererMaterial>& materials);

	void set_camera(const Camera& camera);
	void translate_camera_view(glm::vec3 translation);
	void rotate_camera_view(glm::vec3 rotation_angles);
	void zoom_camera_view(float offset);

	int get_sample_number();
	void set_sample_number(int sample_numner);

	int m_render_width, m_render_height;

	Camera m_camera;

private:
	void set_hiprt_scene_from_scene(Scene& scene);

	OrochiBuffer<Color> m_pixels_buffer;
	OrochiBuffer<hiprtFloat3> m_normals_buffer;
	OrochiBuffer<Color> m_albedo_buffer;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_orochi_ctx;
	oroFunction m_trace_kernel;

	HIPRTScene m_hiprt_scene;
	// The materials are also kept on the CPU side because we want to be able
	// to modify them interactively with ImGui
	std::vector<RendererMaterial> m_materials;

	RenderSettings m_render_settings;
	HIPRTRenderData m_scene_data;
};

#endif

