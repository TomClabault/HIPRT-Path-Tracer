/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/OrochiEnvmap.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HIPRT-Orochi/HIPRTScene.h"
#include "HostDeviceCommon/RenderData.h"
#include "OpenGL/OpenGLInteropBuffer.h"
#include "Renderer/OpenImageDenoiser.h"
#include "Scene/Camera.h"
#include "Scene/SceneParser.h"

#include <vector>

class GPURenderer
{
public:
	GPURenderer();

	void render();
	void change_render_resolution(int new_width, int new_height);

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> get_color_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> get_denoised_framebuffer();
	OrochiBuffer<ColorRGB>& get_denoiser_albedo_buffer();
	OrochiBuffer<hiprtFloat3>& get_denoiser_normals_buffer();
	OrochiBuffer<int>& get_pixels_sample_count_buffer();
	OrochiBuffer<unsigned char>& get_ray_active_buffer();
	OrochiBuffer<unsigned int>& get_stop_noise_threshold_buffer();

	HIPRTRenderSettings& get_render_settings();
	WorldSettings& get_world_settings();
	HIPRTRenderData get_render_data();

	void initialize(int device_index);
	void compile_trace_kernel(const char* kernel_file_path, const char* kernel_function_name);
	void launch_kernel(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args);

	void set_scene(const Scene& scene);
	void set_envmap(ImageRGBA& envmap);
	void set_camera(const Camera& camera);

	const std::vector<RendererMaterial>& get_materials();
	void update_materials(std::vector<RendererMaterial>& materials);

	void translate_camera_view(glm::vec3 translation);
	void rotate_camera_view(glm::vec3 rotation_angles);
	void zoom_camera_view(float offset);

	oroDeviceProp get_device_properties();
	float get_frame_time();
	int get_sample_number();
	void set_sample_number(int sample_numner);

	int m_render_width = 0, m_render_height = 0;

	Camera m_camera;

private:
	void set_hiprt_scene_from_scene(const Scene& scene);

	// Properties of the device
	oroDeviceProp m_device_properties = {};
	// GPU events to time the frame
	oroEvent_t m_frame_start_event, m_frame_stop_event;
	// Time taken to render the last frame
	float m_frame_time = 0;


	// This buffer holds the * sum * of the samples computed
	// This is an accumulation buffer. This needs to be divided by the
	// number of samples for displaying
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> m_framebuffer;
	// Buffer for holding the denoised frame (the denoiser data will be copied
	// to this buffer and then displayed to the viewport)
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> m_denoised_framebuffer;
	// Normals G-buffer
	OrochiBuffer<hiprtFloat3> m_normals_buffer;
	// Albedo G-buffer
	OrochiBuffer<ColorRGB> m_albedo_buffer;

	// Used to calculate the variance of each pixel for adaptive sampling
	OrochiBuffer<float> m_pixels_squared_luminance;
	// This buffer is necessary because with adaptive sampling, each pixel
	// can have accumulated a different number of sample
	OrochiBuffer<int> m_pixels_sample_count;
	// A single boolean to indicate whether there is still a ray active in
	// the kernel or not. Mostly useful when adaptive sampling is on and we
	// want to know if all pixels have converged or not yet
	OrochiBuffer<unsigned char> m_still_one_ray_active_buffer;
	// How many pixels have reached the render_settings.stop_noise_threshold
	OrochiBuffer<unsigned int> m_stop_noise_threshold_count_buffer;

	// The materials are also kept on the CPU side because we want to be able
	// to modify them interactively with ImGui
	std::vector<RendererMaterial> m_materials;
	// Vector to keep the textures data alive otherwise the OrochiTexture objects would
	// be destroyed which means that the underlying textures would be destroyed
	std::vector<OrochiTexture> m_materials_textures;
	OrochiEnvmap m_envmap;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_orochi_ctx;
	// Path tracing kernel called at each frame
	oroFunction m_trace_kernel = nullptr;

	// Structure containing the data specific to a scene:
	//	- hiprtGeom
	//	- hiprtMesh
	//	- materials buffer
	//	- materials indices
	// ...
	//
	// Destroying this structure frees the resources
	HIPRTScene m_hiprt_scene;

	// Settings relative to the scene such as the intensity of the uniform light, the
	// environment map used, the rotation of the envmap, ...
	WorldSettings m_world_settings;
	// Settings that alter the way the path tracing kernel behaves such as the number
	// of bounces, the number of samples per kernel invocation (samples per frame),
	// whether or not the adaptive sampling is enabled, ...
	HIPRTRenderSettings m_render_settings;
};

#endif
