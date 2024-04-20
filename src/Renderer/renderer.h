/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_H
#define RENDERER_H

#include "glm/gtc/matrix_transform.hpp"
#include "HIPRT-Orochi/orochi_buffer.h"
#include "HIPRT-Orochi/hiprt_orochi_ctx.h"
#include "HIPRT-Orochi/hiprt_scene.h"
#include "HostDeviceCommon/render_data.h"
#include "OpenGLInterop/OpenGLInteropBuffer.h"
#include "Scene/camera.h"
#include "Scene/scene_parser.h"

#include "open_image_denoiser.h"

#include <vector>

class Renderer
{
public:
	Renderer(int width, int height, HIPRTOrochiCtx* hiprt_orochi_ctx) : 
		m_render_width(width), m_render_height(height), m_hiprt_orochi_ctx(hiprt_orochi_ctx),
		m_trace_kernel(nullptr)
	{
		m_hiprt_scene.hiprt_ctx = hiprt_orochi_ctx->hiprt_ctx;
	}

	Renderer() : m_hiprt_scene(nullptr) {}

	void render();
	void change_render_resolution(int new_width, int new_height);

	OrochiBuffer<Color>& get_color_framebuffer();
	OrochiBuffer<Color>& get_denoiser_albedo_buffer();
	OrochiBuffer<hiprtFloat3>& get_denoiser_normals_buffer();

	HIPRTRenderSettings& get_render_settings();
	WorldSettings& get_world_settings();
	HIPRTRenderData get_render_data();

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

	// This buffer holds the * sum * of the samples computed
	// This is an accumulation buffer. This needs to be divided by the
	// number of samples for displaying
	OrochiBuffer<Color> m_pixels_buffer;
	// Normals G-buffer
	OrochiBuffer<hiprtFloat3> m_normals_buffer;
	// Albedo G-buffer
	OrochiBuffer<Color> m_albedo_buffer;

	// Used to calculate the variance of each pixel for adaptative sampling
	OrochiBuffer<float> m_pixels_squared_luminance;
	// This buffer is necessary because with adaptative sampling, each pixel
	// can have accumulated a different number of sample
	OrochiBuffer<int> m_pixels_sample_count;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_orochi_ctx;
	oroFunction m_trace_kernel;

	HIPRTScene m_hiprt_scene;
	// The materials are also kept on the CPU side because we want to be able
	// to modify them interactively with ImGui
	std::vector<RendererMaterial> m_materials;

	WorldSettings m_world_settings;
	HIPRTRenderSettings m_render_settings;
};

#endif

