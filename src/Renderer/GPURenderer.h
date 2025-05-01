/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
#include "Renderer/GPUDataStructures/DenoiserBuffersGPUData.h"
#include "Renderer/GPUDataStructures/NEEPlusPlusGPUData.h"
#include "Renderer/GPUDataStructures/StatusBuffersGPUData.h"
#include "Renderer/GPURendererThread.h"
#include "Renderer/HardwareAccelerationSupport.h"
#include "Renderer/OpenImageDenoiser.h"
#include "Renderer/RendererAnimationState.h"
#include "Renderer/RendererEnvmap.h"
#include "Renderer/RenderPasses/GMoNRenderPass.h"
#include "Renderer/RenderPasses/RenderGraph.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRDIRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRGIRenderPass.h"
#include "Renderer/StatusBuffersValues.h"
#include "Scene/Camera.h"
#include "Scene/CameraAnimation.h"
#include "Scene/SceneParser.h"
#include "UI/ApplicationSettings.h"
#include "UI/PerformanceMetricsComputer.h"

#include <unordered_map>
#include <vector>

class RenderWindow;

template <typename T>
class OpenGLInteropBuffer;

class GPURenderer
{
public:
	/**
	 * These constants here are used to reference kernel objects in the 'm_kernels' map
	 * or in the 'm_render_pass_times' map
	 */
	static const std::string NEE_PLUS_PLUS_CACHING_PREPASS_ID;
	static const std::string NEE_PLUS_PLUS_FINALIZE_ACCUMULATION_ID;

	// List of compiler options that will be specific to each kernel. We don't want these options
	// to be synchronized between kernels
	static const std::unordered_set<std::string> KERNEL_OPTIONS_NOT_SYNCHRONIZED;

	// Key for indexing m_render_pass_times that contains the times per passes
	// This key is for the time of the whole frame
	static const std::string ALL_RENDER_PASSES_TIME_KEY;
	static const std::string FULL_FRAME_TIME_WITH_CPU_KEY;
	static const std::string DEBUG_KERNEL_TIME_KEY;

	/**
	 * Constructs a renderer that will be using the given HIPRT/Orochi
	 * context for handling GPU acceleration structures, buffers, textures, etc...
	 */
	GPURenderer(std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx, std::shared_ptr<ApplicationSettings> application_settings);
	~GPURenderer();

	void start_render_thread();

	void setup_brdfs_data();

	/**
	 * Initializes and uploads the fitted parameters for the LTC sheen lobe
	 * of the Principled BSDF
	 */
	void init_sheen_ltc_texture();

	/**
 	 * Initializes the precomputed texture used for GGX energy conservation
	 */
	void load_GGX_energy_compensation_textures(hipTextureFilterMode filtering_mode = hipFilterModeLinear);

	/**
	 * Initializes the precomputed texture used for glossy dielectrics 
	 * energy conservation
	 */
	void load_glossy_dielectric_energy_compensation_textures(hipTextureFilterMode filtering_mode = hipFilterModePoint);

	/**
	 * Initializes the precomputed textures used for GGX glass BSDF energy conservation
	 */
	void load_GGX_glass_energy_compensation_textures(hipTextureFilterMode filtering_mode = hipFilterModePoint);

	/**
	 * Sets up the bounds of the scene for the grid of NEE++
	 */
	void setup_nee_plus_plus_from_scene(const Scene& scene);

	/**
	 * Clears the visibility map of NEE++ so that it is recomputed next frame
	 */
	void reset_nee_plus_plus();

	/**
	 * Computes the alias table for sampling emissive triangles according to power
	 */
	void compute_emissives_power_alias_table(const Scene& scene);
	/**
	 * Overload for computing the alias table at runtime.
	 * This will read the data from the GPU and is thus slower than the overload with the 'scene' parameter
	 * 
	 * This function is mainly used because at runtime, we don't have the scene data anymore since it's been freed
	 * from the CPU to save on RAM
	 */
	void recompute_emissives_power_alias_table();
	void free_emissives_power_alias_table();
	bool needs_emissives_power_alias_table();

	std::shared_ptr<GMoNRenderPass> get_gmon_render_pass();
	std::shared_ptr<ReGIRRenderPass> get_ReGIR_render_pass();
	std::shared_ptr<ReSTIRDIRenderPass> get_ReSTIR_DI_render_pass();
	std::shared_ptr<ReSTIRGIRenderPass> get_ReSTIR_GI_render_pass();

	NEEPlusPlusGPUData& get_nee_plus_plus_data();

	/**
	 * Initializes the filter function used by the kernels
	 */
	void setup_filter_functions();

	/**
	 * Steps all the animations of the renderer one step forward
	 * 
	 * The 'delta_time' parameter should be how much time passed, in milliseconds, since the last
	 * call to step_animations()
	 */
	void step_animations(float delta_time);

	/**
	 * Blocking that waits for all the operations queued on
	 * the main stream to complete
	 */
	void synchronize_all_kernels();

	/**
	 * Returns true if the last frame was rendered with render_settings.wants_render_low_resolution = true.
	 * False otherwise
	 */
	bool was_last_frame_low_resolution();

	/**
	 * Whether or not the current frame has finished rendering
	 */
	bool frame_render_done();

	/**
	 * Resizes all the buffers of the renderer to the given new width and height
	 */
	void resize(int new_width, int new_height);

	/**
	 * Requests a frame to the render thread
	 */
	void render(float delta_time_gpu, RenderWindow* render_window);

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
	 *
	 * The 'delta_time' parameter should be how much time passed, in milliseconds, since the last
	 * call to pre_render_update()
	 */
	void pre_render_update(float delta_time, RenderWindow* render_window);

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

	/**
	 * Switches to using OpenGL interop buffers or classical non-interop CUDA/HIP buffers for the denoiser AOVs.
	 * 
	 * OpenGL Interop buffers seems to be slower and slow down the path tracing kernel on AMD for some reasons.
	 * Using non-interop buffers thus increases rendering performance but tanks denoising performance
	 */
	void set_use_denoiser_AOVs_interop_buffers(bool use_interop);

	/**
	 * Returns the framebuffer that should be used for displaying to the viewport
	 * 
	 * At the time of writing this comment, this is either the default framebuffer where the
	 * ray colors are accumulated or the GMoN result framebuffer where the median of means are
	 * computed for fireflies reduction
	 */
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_color_interop_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_default_interop_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_denoised_interop_framebuffer();
	std::shared_ptr<OpenGLInteropBuffer<float3>> get_denoiser_normals_AOV_interop_buffer();
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_denoiser_albedo_AOV_interop_buffer();
	std::shared_ptr<OrochiBuffer<float3>> get_denoiser_normals_AOV_no_interop_buffer();
	std::shared_ptr<OrochiBuffer<ColorRGB32F>> get_denoiser_albedo_AOV_no_interop_buffer();
	std::shared_ptr<OrochiBuffer<int>>& get_pixels_converged_sample_count_buffer();
	/**
	 * Returns a structure that contains the values of
	 * various one-variable buffers of the renderer such
	 * as 'one_ray_active' or 'pixel_converged_count' for example
	 */
	const StatusBuffersValues& get_status_buffer_values() const;
	/**
	 * Memcpy the values of the status buffers to m_status_buffer_values
	 */
	void download_status_buffers();

	HIPRTRenderSettings& get_render_settings();
	std::shared_ptr<ApplicationSettings> get_application_settings();
	WorldSettings& get_world_settings();
	HIPRTRenderData& get_render_data();
	HIPRTScene& get_hiprt_scene();
	std::shared_ptr<HIPRTOrochiCtx> get_hiprt_orochi_ctx();
	void invalidate_render_data_buffers();

	Camera& get_camera();
	Camera& get_previous_frame_camera();
	CameraAnimation& get_camera_animation();
	RendererEnvmap& get_envmap();
	SceneMetadata& get_scene_metadata();
	RenderGraph& get_render_graph();

	void set_scene(const Scene& scene);
	void rebuild_renderer_bvh(hiprtBuildFlags build_flags, bool do_compaction);
	void set_camera(const Camera& camera);
	void set_envmap(const Image32Bit& envmap, const std::string& envmap_filepath);
	bool has_envmap();

	const std::vector<CPUMaterial>& get_original_materials();
	const std::vector<CPUMaterial>& get_current_materials();
	const std::vector<std::string>& get_material_names();

	/**
	 * Updates all the materials of the renderer and reuploads them all to the GPU
	 */
	void update_all_materials(std::vector<CPUMaterial>& materials);
	/**
	 * Updates only the material with index 'material_index' and uploads it to the GPU 
	 */
	void update_one_material(CPUMaterial& material, int material_index);

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
	 * the RayVolumeState struct includes a NestedDielectricsInteriorStack struct whose size
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
	// size_t get_ray_volume_state_byte_size();

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

	/**
	 * Returns a map of all the kernels of the renderer
	 * 
	 * The map keys are the kernel name
	 * The map values are the kernel themselves
	 */
	std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels();
	/**
	 * Returns a map of all the kernels of the renderer that trace rays (shadow rays, bounce rays, ...)
	 * 
	 * This is used in ImGui in the performance settings panel where we can adjust the
	 * amount of shared memory used for the BVH traversal. Because this is only useful for
	 * kernels that trace rays, we want a function that returns only the kernels that trace rays
	 *
	 * The map keys are the kernel name
	 * The map values are the kernel themselves
	 */
	std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels();

	//std::vector<std::string> get_all_kernel_ids();
	/**
	 * Sets the debug kernel to be used.
	 * 
	 * The kernel is expected to be in a file called {kernel_name}.h and the entry point
	 * function is expected to be {kernel_name}
	 * 
	 * Calling this function with an empty string as parameter clears the debug kernel
	 */
	void set_debug_trace_kernel(const std::string& kernel_name, GPUKernelCompilerOptions options = GPUKernelCompilerOptions());
	bool is_using_debug_kernel();
	oroStream_t get_main_stream();
	oroStream_t get_async_stream_1();

	std::unordered_map<std::string, float>& get_render_pass_times();
	/**
	 * Returns the time taken to compute the last frame in milliseconds
	 */
	float get_last_frame_time();

	void update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics);

	void reset(bool reset_by_camera_movement);

	Xorshift32Generator& get_rng_generator();

	int2 m_render_resolution = make_int2(0, 0);

	Camera m_camera;
	Camera m_previous_frame_camera;
	// Animator of the camera of the current frame ('m_camera')
	CameraAnimation m_camera_animation;

	OrochiBuffer<unsigned long long int> m_DEBUG_SUM_COUNT;
	OrochiBuffer<float> m_DEBUG_SUMS;

private:
	// So that GPURendererThread can access the private members of GPURenderer without
	// having to write a thousand getters that would expose many members to everyone else
	friend class GPURendererThread;

	void set_hiprt_scene_from_scene(const Scene& scene);
	void update_render_data();

	/**
	 * Private function that does the actual alias table recomputation
	 */
	void compute_emissives_power_alias_table(
		const std::vector<int>& emissive_triangle_indices,
		const std::vector<float3>& vertices_positions,
		const std::vector<int>& triangles_indices,
		const std::vector<int>& material_indices,
		const std::vector<CPUMaterial>& materials,

		OrochiBuffer<float>& alias_table_probas_buffer,
		OrochiBuffer<int>& alias_table_alias_buffer,
		DeviceAliasTable& power_alias_table);

	/**
	 * Returns true if one of the kernels requires the global stack buffer for BVH traversal
	 */
	bool needs_global_bvh_stack_buffer();
	/**
	 * Frees and recreates the global stack buffer for the BVH traversal based on the
	 * current resolution of the renderer
	 */
	void recreate_global_bvh_stack_buffer();

	/**
	 * Reads the execution time of the kernels and stores those execution times in 'm_render_pass_times'
	 */
	void compute_render_pass_times();

	void launch_nee_plus_plus_caching_prepass();

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

	// ---- Functions called by the pre_render_update() method ----
	//

	//
	// -------- Functions called by the pre_render_update() method ---------

	void internal_clear_m_status_buffers();

	GPURendererThread m_render_thread;

	// Some render passes want the application settings of the render window so it's there
	std::shared_ptr<ApplicationSettings> m_application_settings;

	// Properties of the device
	oroDeviceProp m_device_properties = { .gcnArchName = "" };

	// GPU events to time the frame
	/*oroEvent_t m_frame_start_event = nullptr;
	oroEvent_t m_frame_stop_event = nullptr;*/
	// If true, the last call to render() rendered a frame where render_settings.render_low_resoltion was true.
	// False otherwise
	bool m_was_last_frame_low_resolution = false;
	// If true, the buffer pointers of m_render_data will be updated when pre_render_update() is called.
	// This boolean is mainly set to true when resizing the renderer since resizing re-creates the 
	// buffers -> invalidates the pointer -> we need to set them back on render_data
	//
	// Modifying the scene also invalidates the m_render_data buffers. 
	// Freeing / allocating ReSTIR DI/adaptive sampling buffers (or any buffers that can be allocated / dealloacted) too
	bool m_render_data_buffers_invalidated = true;
	// Whether or not the renderer was updated (with pre_render_update()) since the last render() call.
	// This is only used as a security to avoid misusing the renderer class and calling render()
	// without having called pre_render_update() before
	bool m_updated = false;

	// Time taken per each pass of the renderer for the last frame.
	// 
	// Some more keys are defined as static const std::string members of this class
	std::unordered_map<std::string, float> m_render_pass_times;

	// This buffer holds the * sum * of the samples computed
	// This is an accumulation buffer. This needs to be divided by the
	// number of samples for displaying
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> m_framebuffer;
	// AOVs and buffers needed by the denoiser that are filled/manipulated by the renderer.
	// This is just a structure to aggregate them all instead of having multiple
	// variables in the renderer class
	DenoiserBuffersGPUData m_denoiser_buffers;

	// Used to calculate the variance of each pixel for adaptive sampling
	OrochiBuffer<float> m_pixels_squared_luminance_buffer;
	// This buffer stores the number of samples accumulated *until* a pixel has converged
	// ("converged" is according to adaptive sampling or pixel stop noise threshold)
	std::shared_ptr<OrochiBuffer<int>> m_pixels_converged_sample_count_buffer;
	// This buffer is necessary because with adaptive sampling, each pixel
	// can have accumulated a different number of sample
	OrochiBuffer<int> m_pixels_sample_count_buffer;
	OrochiBuffer<unsigned char> m_pixel_active;

	StatusBuffersGPUData m_status_buffers;
	// Whether or not the pixel at the given index is active and needs more samples
	// Structure that holds the values of the one-variable buffers of the renderer.
	// These values are 'one_ray_active' or 'pixel_converged_count' for example.
	// These values are updated when the pre_render_update() is called
	StatusBuffersValues m_status_buffers_values;

	// Some additional info about the parsed scene such as materials names, mesh names, ...
	SceneMetadata m_parsed_scene_metadata;
	// The original materials of the scene. Those are the materials that have directly been read from the hard drive scene file.
	// Used in case the user wants to revert every changes that have been done
	std::vector<CPUMaterial> m_original_materials;
	// Materials currently being used by the GPU. Those are the materials *currently* being
	// used for rendering
	std::vector<CPUMaterial> m_current_materials;
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
	// See 'setup_render_passes' for more details on how that "synchronization" is setup
	std::shared_ptr<GPUKernelCompilerOptions> m_global_compiler_options;

	// Render passes/kernels used for the ray tracing
	// They are all organized in a map so that we can iterate over them. The key
	// of this map is a "name"
	std::map<std::string, std::shared_ptr<GPUKernel>> m_kernels;

	// Additional functions called on hits when tracing rays (alpha testing for example)
	std::vector<hiprtFuncNameSet> m_func_name_sets;

	// HIPRT and Orochi contexts
	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_orochi_ctx = nullptr;

	// Custom stream onto which kernels are dispatched
	oroStream_t m_main_stream = nullptr;

	// Buffers and settings for NEE++
	NEEPlusPlusGPUData m_nee_plus_plus;
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

	// Precomputed tables for GGX energy compensation
	// [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
	OrochiTexture m_GGX_conductor_directional_albedo;
	OrochiTexture3D m_glossy_dielectric_directional_albedo;
	OrochiTexture3D m_GGX_glass_directional_albedo;
	OrochiTexture3D m_GGX_glass_inverse_directional_albedo;
	OrochiTexture3D m_GGX_thin_glass_directional_albedo;
};

#endif
