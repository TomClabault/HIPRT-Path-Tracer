/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include "Compiler/GPUKernel.h"
#include "Device/kernel_parameters/ReSTIR/DI/LightPresamplingParameters.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/OrochiTexture3D.h"
#include "HIPRT-Orochi/HIPRTScene.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/RendererAnimationState.h"
#include "Renderer/RendererEnvmap.h"
#include "Renderer/GPURendererGBuffer.h"
#include "Renderer/HardwareAccelerationSupport.h"
#include "Renderer/OpenImageDenoiser.h"
#include "Renderer/StatusBuffersValues.h"
#include "Renderer/RenderPasses/ReSTIRDIRenderPass.h"
#include "Scene/Camera.h"
#include "Scene/CameraAnimation.h"
#include "Scene/SceneParser.h"
#include "UI/ApplicationSettings.h"
#include "UI/PerformanceMetricsComputer.h"

#include <unordered_map>
#include <vector>

template <typename T>
class OpenGLInteropBuffer;

class GPURenderer
{
public:
	/**
	 * These constants here are used to reference kernel objects in the 'm_kernels' map
	 * or in the 'm_render_pass_times' map
	 */
	static const std::string CAMERA_RAYS_KERNEL_ID;
	static const std::string PATH_TRACING_KERNEL_ID;
	static const std::string RAY_VOLUME_STATE_SIZE_KERNEL_ID;

	// List of compiler options that will be specific to each kernel. We don't want these options
	// to be synchronized between kernels
	static const std::unordered_set<std::string> KERNEL_OPTIONS_NOT_SYNCHRONIZED;

	/**
	 * This map contains constants that are the name of the main function of the kernels, their entry points.
	 * They are used when compiling the kernels.
	 * 
	 * This means that if you define your camera ray kernel main function as:
	 * 
	 * GLOBAL_KERNEL_SIGNATURE(void) CameraRays(HIPRTRenderData render_data, int2 res)
	 * 
	 * Then KERNEL_FUNCTION_NAMES[CAMERA_RAYS_KERNEL_ID] = "CameraRays"
	 */
	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;

	/**
	 * Same as 'KERNELfUNCTION_NAMES' but for kernel files
	 */
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	// Key for indexing m_render_pass_times that contains the times per passes
	// This key is for the time of the whole frame
	static const std::string FULL_FRAME_TIME_KEY;

	/**
	 * Constructs a renderer that will be using the given HIPRT/Orochi
	 * context for handling GPU acceleration structures, buffers, textures, etc...
	 */
	GPURenderer(std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx);
	void setup_brdfs_data();

	/**
	 * Initializes and uploads the fitted parameters for the LTC sheen lobe
	 * of the Principled BSDF
	 */
	void init_sheen_ltc_texture();

	/**
 	 * Initializes the precomputed texture used for GGX energy conservation
	 */
	void init_GGX_Ess_texture(HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_LINEAR);

	/**
	 * Initializes the precomputed texture used for glossy dielectrics 
	 * energy conservation
	 */
	void init_glossy_dielectric_Ess_texture(HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_LINEAR);

	/**
	 * Initializes the precomputed textures used for GGX glass BSDF energy conservation
	 */
	void init_GGX_glass_Ess_texture(HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_POINT);

	/**
	 * Initializes the filter function used by the kernels
	 */
	void setup_filter_functions();

	/**
	 * Initializes and compiles the kernels
	 */
	void setup_kernels();

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
	 * Returns true if one of the kernels requires the global stack buffer for BVH traversal
	 */
	bool needs_global_bvh_stack_buffer();

	/**
	 * Renders a frame asynchronously. 
	 * Querry frame_render_done() to know whether or not the frame has completed or not.
	 */
	void render();

	void launch_camera_rays();
	void launch_ReSTIR_DI();
	void launch_path_tracing();

	/**
	 * Blocking that waits for all the operations queued on
	 * the main stream to complete
	 */
	void synchronize_kernel();

	/**
	 * Returns false if the frame queued asynchronously by a previous call to render()
	 * isn't finished yet. 
	 * Returns true if the frame is completed
	 */
	bool frame_render_done();
	/**
	 * Returns true if the last frame was rendered with render_settings.wants_render_low_resolution = true.
	 * False otherwise
	 */
	bool was_last_frame_low_resolution();

	/**
	 * Resizes all the buffers of the renderer to the given new width and height
	 * 
	 * If 'also_resize_interop' is true, OpenGL Interop buffers will also be resized
	 * by this function call. Resizing OpenGL Interop buffers cannot be done on a
	 * thread other than the main thread so if resizing the renderer asynchronously 
	 * on multiple threads, 'also_resize_interop' needs to be passed as false and 
	 * GPURenderer::resize_interop_buffers() must then be called on the main thread
	 */
	void resize(int new_width, int new_height, bool also_resize_interop = true);
	/**
	 * Resizes only OpenGL Interop Buffers. Useful to resize the renderer on
	 * a separate thread because OpenGL interop buffers cannot be resized on a
	 * separate thread so we need to resize them on the main thread,
	 * using this function
	 */
	void resize_interop_buffers(int new_width, int new_height);

	/**
	 * Maps the buffers shared with OpenGL that are needed for rendering the frame and sets
	 * their mapped pointer into m_render_data
	 */
	void map_buffers_for_render();

	/**
	 * Unmap the color framebuffer, the denoiser albedo and the
	 * denoiser normals buffers so that OpenGL can use them
	 */
	void unmap_buffers();

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_color_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_denoised_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<float3>> get_denoiser_normals_AOV_buffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_denoiser_albedo_AOV_buffer();
	std::shared_ptr<OpenGLInteropBuffer<int>>& get_pixels_converged_sample_count_buffer();
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
	HIPRTRenderData& get_render_data();
	HIPRTScene& get_hiprt_scene();
	std::shared_ptr<HIPRTOrochiCtx> get_hiprt_orochi_ctx();
	void invalidate_render_data_buffers();

	Camera& get_camera();
	CameraAnimation& get_camera_animation();
	RendererEnvmap& get_envmap();

	void set_scene(const Scene& scene);
	void set_camera(const Camera& camera);
	void set_envmap(const Image32Bit& envmap, const std::string& envmap_filepath);
	bool has_envmap();

	const std::vector<RendererMaterial>& get_original_materials();
	const std::vector<RendererMaterial>& get_current_materials();
	const std::vector<std::string>& get_material_names();
	void update_materials(std::vector<RendererMaterial>& materials);

	const std::vector<BoundingBox>& get_mesh_bounding_boxes();
	const std::vector<std::string>& get_mesh_names();
	const std::vector<int>& get_mesh_material_indices();

	/**
	 * Returns the size of the RayVolumeState struct on the GPU.
	 * 
	 * Useful when the size of the struct changes because the nested dielectrics
	 * stack size changed but we have no easy way to find out what's the new size
	 * of the struct on the CPU to upload the correct data size.
	 * 
	 * There's no easy way to find the new size of the struct on the CPU because
	 * the RayVolumeState struct includes a InteriorStackImpl struct whose size
	 * is defined at compilation time. If the nested dielectrics stack size changes
	 * at runtime (possible through ImGui), then we need to recompute the size of
	 * the RayVolumeState structure on the CPU to be able to properly resize the
	 * GPU buffers that use the RayVolumeState (in the GBuffer for example).
	 * However, again, that size is determined at compilation time so we can't
	 * know on the CPU what's going to be the new size. To circumvent that, we
	 * use the fact that shader are recompiled on the GPU and so the shaders know
	 * the new size. This function thus launches a kernel on the GPU to querry
	 * the size of the structure.
	 */
	size_t get_ray_volume_state_byte_size();

	/**
	 * Resizes the ray_volume_states array of the GBuffers
	 * (current frame and previous frames if used) so that it matches the size of RayVolumeState being used on the GPU
	 */
	void resize_g_buffer_ray_volume_states();

	void translate_camera_view(glm::vec3 translation);
	/**
	 * Rotates the camera by the given angles (in radians)
	 */
	void rotate_camera_view(glm::vec3 rotation_angles);
	void zoom_camera_view(float offset);

	RendererAnimationState& get_animation_state();

	oroDeviceProp get_device_properties();
	HardwareAccelerationSupport device_supports_hardware_acceleration();

	std::shared_ptr<GPUKernelCompilerOptions> get_global_compiler_options();

	void recompile_kernels(bool use_cache = true);
	void take_kernel_compilation_priority();
	void release_kernel_compilation_priority();
	/**
	 * Precompiles a variety of kernel option combinations to avoid
	 * having to compile too many kernels at runtime
	 */
	void precompile_kernels();
	void stop_background_shader_compilation();
	void resume_background_shader_compilation();

	std::map<std::string, GPUKernel*> get_kernels();
	oroStream_t get_main_stream();

	void compute_render_pass_times();
	std::unordered_map<std::string, float>& get_render_pass_times();
	/**
	 * Returns the time taken to compute the last frame in milliseconds
	 */
	float get_last_frame_time();

	void update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics);

	void reset(std::shared_ptr<ApplicationSettings> application_settings);

	Xorshift32Generator& rng();

	int2 m_render_resolution = make_int2(0, 0);

	Camera m_camera;
	Camera m_previous_frame_camera;
	// Animator of the camera of the current frame ('m_camera')
	CameraAnimation m_camera_animation;

private:
	void set_hiprt_scene_from_scene(const Scene& scene);
	void update_render_data();

	/**
	 * Precompiles direct lighting strategy kernels
	 */
	void precompile_direct_light_sampling_kernels();
	/**
	 * Precompiles ReSTIR DI kernels
	 */
	void precompile_ReSTIR_DI_kernels();
	/**
	 * Precompiles a single kernel given its ID and the options
	 * that will be overriden when compiling the kernel
	 */
	void precompile_kernel(const std::string& id, GPUKernelCompilerOptions partial_options);

	// ---- Functions called by the update() method ----
	//

	/**
	 * Resets the value of the status buffers on the device
	 */
	void internal_update_clear_device_status_buffers();

	/**
	 * Allocates/deallocates the G-buffer of the previous frame depending
	 * on whether or not it is needed
	 */
	void internal_update_prev_frame_g_buffer();

	/**
	 * This function evaluates whether the renderer needs the adaptive
	 * sampling buffers or not. If the buffers are needed (because the
	 * adaptive sampling or the stop noise pixel threshold is enabled for example),
	 * then the buffer will be allocated so that they can be used by the shader.
	 * If they are not needed, they will be freed to save some VRAM.
	 */
	void internal_update_adaptive_sampling_buffers();

	/**
	 * Allocates/frees the global buffer for BVH traversal when UseSharedStackBVHTraversal is TRUE
	 */
	void internal_update_global_stack_buffer();

	//
	// -------- Functions called by the update() method ---------

	void internal_clear_m_status_buffers();

	// Properties of the device
	oroDeviceProp m_device_properties = { .gcnArchName = "" };

	// GPU events to time the frame
	oroEvent_t m_frame_start_event = nullptr;
	oroEvent_t m_frame_stop_event = nullptr;
	// If true, the last call to render() rendered a frame where render_settings.render_low_resoltion was true.
	// False otherwise
	bool m_was_last_frame_low_resolution = false;
	// If true, the buffer pointers of m_render_data will be updated when update() is called.
	// This boolean is mainly set to true when resizing the renderer since resizing re-creates the 
	// buffers -> invalidates the pointer -> we need to set them back on render_data
	//
	// Modifying the scene also invalidates the m_render_data buffers. 
	// Freeing / allocating ReSTIR DI/adaptive sampling buffers (or any buffers that can be allocated / dealloacted) too
	bool m_render_data_buffers_invalidated = true;

	// Time taken per each pass of the renderer. 
	// An additional key GPURenderer::FULL_FRAME_TIME_KEY can be used to index in this map
	// and retrieve the time for the whole frame
	std::unordered_map<std::string, float> m_render_pass_times;

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

	// G-buffers of the current frame (camera rays hits) and previous frame
	GPURendererGBuffer m_g_buffer;
	GPURendererGBuffer m_g_buffer_prev_frame;

	// Used to calculate the variance of each pixel for adaptive sampling
	OrochiBuffer<float> m_pixels_squared_luminance_buffer;
	// This buffer stores the number of samples accumulated *until* a pixel has converged
	// ("converged" is according to adaptive sampling or pixel stop noise threshold)
	std::shared_ptr<OpenGLInteropBuffer<int>> m_pixels_converged_sample_count_buffer;
	// This buffer is necessary because with adaptive sampling, each pixel
	// can have accumulated a different number of sample
	OrochiBuffer<int> m_pixels_sample_count_buffer;
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

	ReSTIRDIRenderPass m_restir_di_render_pass;

	// Some additional info about the parsed scene such as materials names, mesh names, ...
	SceneMetadata m_parsed_scene_metadata;
	// The original materials of the scene. Those are the materials that have directly been read from the hard drive scene file.
	// Used in case the user wants to revert every changes that have been done
	std::vector<RendererMaterial> m_original_materials;
	// Materials currently being used by the GPU. Those are the materials *currently* being
	// used for rendering
	std::vector<RendererMaterial> m_current_materials;
	// The material names are used for displaying in the ImGui editor
	// AABB of the meshes of the scene
	std::vector<BoundingBox> m_mesh_bounding_boxes;

	// Options used for compiling the render passes of this renderer.
	// 
	// Most of the options in there are shared with all the passes. For example,
	// the "__USE_HWI__" macro that dictates whether to use hardware acceleration
	// ray tracing is shared between all kernels (because there's no real reasons for 
	// one kernel not to use it if all other kernels use it).
	// The value 1 or 0 of this macro is stored in this 'm_global_compiler_options' member
	// and is 'synchronized' through the use of pointers with the options of the other kernels.
	// See 'setup_kernels' for more details on how that "synchronization" is setup
	std::shared_ptr<GPUKernelCompilerOptions> m_global_compiler_options;

	// Render passes/kernels used for the ray tracing
	// They are all organized in a map so that we can iterate over them. The key
	// of this map is a "name"
	std::map<std::string, GPUKernel> m_kernels;
	// Whether or not kernel precompilation has been launched yet
	bool m_kernel_precompilation_launched = false;

	// Kernel used for retrieving the size of the RayVolumeState structure on the GPU
	GPUKernel m_ray_volume_state_byte_size_kernel;

	// Additional functions called on hits when tracing rays (alpha testing for example)
	std::vector<hiprtFuncNameSet> m_func_name_sets;

	// HIPRT and Orochi contexts
	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_orochi_ctx = nullptr;

	// Custom stream onto which kernels are dispatched asynchronously
	oroStream_t m_main_stream= nullptr;

	// Render data passed to the GPU for rendering. Most importantly it contains
	// 
	// The WorldSettings: Settings relative to the scene such as the intensity of the uniform light, the
	// environment map used, the rotation of the envmap, ...
	// 
	// The RenderSettings: Settings that alter the way the path tracing kernel behaves such as the number
	// of bounces, the number of samples per kernel invocation (samples per frame),
	// whether or not the adaptive sampling is enabled, ...
	HIPRTRenderData m_render_data;

	// Structure containing the data specific to a scene:
	//	- hiprtGeom
	//	- hiprtMesh
	//	- materials buffer
	//	- materials indices
	// ...
	//
	// Destroying this structure frees the resources
	HIPRTScene m_hiprt_scene;

	// Random number generator used to fill the render_data.random_seed argument
	// in update_render_data().
	Xorshift32Generator m_rng;

	// State of the animation of the renderer
	RendererAnimationState m_animation_state;

	// Envmap of the renderer
	RendererEnvmap m_envmap;

	// 32x32 texture containing the precomputed parameters of the LTC
	// fitted to approximate the SSGX sheen volumetric layer.
	// See SheenLTCFittedParameters.h
	OrochiTexture m_sheen_ltc_params;

	// Precomputed tables for GGX energy conservation
	// [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
	OrochiTexture m_GGX_Ess;
	OrochiTexture3D m_glossy_dielectric_Ess;
	OrochiTexture3D m_GGX_Ess_glass;
	OrochiTexture3D m_GGX_Ess_glass_inverse;
};

#endif
