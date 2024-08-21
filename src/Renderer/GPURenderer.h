/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/OrochiEnvmap.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HIPRT-Orochi/HIPRTScene.h"
#include "HostDeviceCommon/RenderData.h"
#include "OpenGL/OpenGLInteropBuffer.h"
#include "Renderer/HardwareAccelerationSupport.h"
#include "Renderer/OpenImageDenoiser.h"
#include "Renderer/StatusBuffersValues.h"
#include "Renderer/ReSTIR/ReSTIR_DI_Reservoirs.h"
#include "Scene/Camera.h"
#include "Scene/SceneParser.h"

#include <vector>

class GPURenderer
{
public:
	static const std::string PATH_TRACING_KERNEL;
	static const std::string CAMERA_RAYS_FUNC_NAME;
	static const std::string RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME;
	static const std::string RESTIR_DI_SPATIAL_REUSE_FUNC_NAME;

	static const std::string KERNEL_FILES[];
	static const std::string KERNEL_FUNCTIONS[];

	static const std::vector<std::string> COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS;

	/**
	 * Constructs a renderer that will be using the given HIPRT/Orochi
	 * context for handling GPU acceleration structures, buffers, textures, etc...
	 */
	GPURenderer(std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx);

	/**
	 * This function is in charge of updating various "dynamic attributes/properties/buffers" of the renderer before rendering a frame.
	 * 
	 * These "dynamic attributes/properties/buffers" can be the adaptive sampling buffers for example.
	 * 
	 * It will be checked each whether or not the adaptive sampling buffers need to be
	 * allocated or freed and action will be taken accordingly. This function basically enables a
	 * nice behavior of the application in which the renderer "automatically" reacts to changes
	 * that could be made (through the ImGui interface for example) so that it is always in the
	 * correct state. Said othrewise, this function can be seen as a centralized place for updating
	 * various stuff of the renderer instead of having to scatter these update calls everywhere
	 * in the code.
	 */
	void update();

	/**
	 * Renders a frame asynchronously. 
	 * Querry frame_render_done() to know whether or not the frame has completed or not.
	 */
	void render();

	/**
	 * Blocking that waits for all the operations queued on
	 * the main stream to complete
	 */
	void synchronize_kernel();

	/**
	 * Returns false if the frame queued asynchronously by a previous call to render() isn't finished yet. 
	 * Returns true if the frame is completed
	 */
	bool frame_render_done();
	/**
	 * Returns true if the last frame was rendered with render_settings.render_low_resolution = true.
	 * False otherwise
	 */
	bool was_last_frame_low_resolution();

	void resize(int new_width, int new_height);

	/**
	 * Unmap the color framebuffer, the denoiser albedo and the
	 * denoiser normals buffers so that OpenGL can use them
	 */
	void unmap_buffers();

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_color_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_denoised_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<float3>> get_denoiser_normals_AOV_buffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_denoiser_albedo_AOV_buffer();
	std::shared_ptr<OpenGLInteropBuffer<int>>& get_pixels_sample_count_buffer();
	/**
	 * Returns a structure that contains the values of
	 * various one-variable buffers of the renderer such
	 * as 'one_ray_active' or 'pixel_converged_count' for example
	 */
	const StatusBuffersValues& get_status_buffer_values() const;
	/**
	 * Memcpy the values of the status buffers to m_status_buffer_values
	 */
	void copy_status_buffers();

	HIPRTRenderSettings& get_render_settings();
	WorldSettings& get_world_settings();
	HIPRTRenderData get_render_data();

	Camera& get_camera();

	void set_scene(const Scene& scene);
	void set_camera(const Camera& camera);
	void set_envmap(Image32Bit& envmap);
	bool has_envmap();

	const std::vector<RendererMaterial>& get_materials();
	const std::vector<std::string>& get_material_names();
	void update_materials(std::vector<RendererMaterial>& materials);

	void translate_camera_view(glm::vec3 translation);
	/**
	 * Rotates the camera by the given angles (in radians)
	 */
	void rotate_camera_view(glm::vec3 rotation_angles);
	void zoom_camera_view(float offset);

	oroDeviceProp get_device_properties();
	HardwareAccelerationSupport device_supports_hardware_acceleration();

	std::shared_ptr<GPUKernelCompilerOptions> get_path_tracer_options();

	void recompile_kernels(bool use_cache = true);

	float get_last_frame_time();
	void reset_last_frame_time();
	void reset();

	int m_render_width = 0, m_render_height = 0;

	Camera m_camera;

private:
	void set_hiprt_scene_from_scene(const Scene& scene);

	// ---- Functions called by the update() method ----
	//

	/**
	 * Resets the value of the status buffers on the device
	 */
	void internal_update_clear_device_status_buffers();

	/**
	 * This function evaluates whether the renderer needs the adaptive
	 * sampling buffers or not. If the buffers are needed (because the
	 * adaptive sampling or the stop noise pixel threshold is enabled for example),
	 * then the buffer will be allocated so that they can be used by the shader.
	 * If they are not needed, they will be freed to save some VRAM.
	 */
	void internal_update_adaptive_sampling_buffers();

	//
	// -------- Functions called by the update() method ---------



	void internal_clear_m_status_buffers();

	// Properties of the device
	oroDeviceProp m_device_properties = { .gcnArchName = "" };

	// GPU events to time the frame
	oroEvent_t m_frame_start_event = nullptr;
	oroEvent_t m_frame_stop_event = nullptr;
	// Time taken to render the last frame
	float m_last_frame_time = 0;
	// If true, the last call to render() rendered a frame where render_settings.render_low_resoltion was true.
	// False otherwise
	bool m_was_last_frame_low_resolution = false;

	// This buffer holds the * sum * of the samples computed
	// This is an accumulation buffer. This needs to be divided by the
	// number of samples for displaying
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> m_framebuffer;
	// Buffer for holding the denoised frame (the denoiser data will be copied
	// to this buffer and then displayed to the viewport)
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> m_denoised_framebuffer;
	// Normals G-buffer
	std::shared_ptr<OpenGLInteropBuffer<float3>> m_normals_AOV_buffer;
	// Albedo G-buffer
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>>m_albedo_AOV_buffer;

	// GBuffer that stores information about the current frame first hit data
	struct GBuffer
	{
		OrochiBuffer<SimplifiedRendererMaterial> materials;

		OrochiBuffer<float3> shading_normals;
		OrochiBuffer<float3> geometric_normals;
		OrochiBuffer<float3> view_directions;
		OrochiBuffer<float3> first_hits;

		OrochiBuffer<unsigned char> cameray_ray_hit;

		OrochiBuffer<RayVolumeState> ray_volume_states;
	};

	GBuffer m_g_buffer;

	// Used to calculate the variance of each pixel for adaptive sampling
	OrochiBuffer<float> m_pixels_squared_luminance_buffer;
	// This buffer is necessary because with adaptive sampling, each pixel
	// can have accumulated a different number of sample
	std::shared_ptr<OpenGLInteropBuffer<int>> m_pixels_sample_count_buffer;
	// A single boolean to indicate whether there is still a ray active in
	// the kernel or not. Mostly useful when adaptive sampling is on and we
	// want to know if all pixels have converged or not yet
	OrochiBuffer<unsigned char> m_still_one_ray_active_buffer;
	// How many pixels have reached the render_settings.stop_pixel_noise_threshold.
	// Warning: This buffer does not count how many pixels have converged according to
	// the adaptive sampling noise threshold. This is only for the stop_pixel_noise_threshold
	OrochiBuffer<unsigned int> m_pixels_converged_count_buffer;
	// Whether or not the pixel at the given index is active and needs more samples
	OrochiBuffer<unsigned char> m_pixel_active;

	// Structure that holds the values of the one-variable buffers of the renderer.
	// These values are 'one_ray_active' or 'pixel_converged_count' for example.
	// These values are updated when the update() is called
	StatusBuffersValues m_status_buffers_values;

	// Various reservoirs used by ReSTIR DI
	ReSTIR_DI_Reservoirs m_restir_di_reservoirs;

	// The materials are also kept on the CPU side because we want to be able
	// to modify them interactively with ImGui
	std::vector<RendererMaterial> m_materials;
	// The material names are used for displaying in the ImGui editor
	std::vector<std::string> m_material_names;
	// Vector to keep the textures data alive otherwise the OrochiTexture objects would
	// be destroyed which means that the underlying textures would be destroyed
	std::vector<OrochiTexture> m_materials_textures;
	OrochiEnvmap m_envmap;

	std::shared_ptr<GPUKernelCompilerOptions> m_path_tracer_options;
	// Path tracing kernel called at each frame
	GPUKernel m_camera_ray_pass;
	GPUKernel m_restir_initial_candidates_pass;
	GPUKernel m_restir_spatial_reuse_pass;
	GPUKernel m_path_trace_pass;
	
	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_orochi_ctx = nullptr;

	oroStream_t m_main_stream;

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

	// Random number generator used to fill the render_data.random_seed argument
	// in get_render_data().
	Xorshift32Generator m_rng;
};

#endif
