	/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Device/includes/BSDFs/SheenLTCFittedParameters.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "Renderer/Baker/GPUBaker.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/GPURenderer.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"

#include <Orochi/OrochiUtils.h>

#include <condition_variable>

const std::string GPURenderer::NEE_PLUS_PLUS_CACHING_PREPASS_ID = "NEE++ Caching Prepass";
const std::string GPURenderer::CAMERA_RAYS_KERNEL_ID = "Camera Rays";
const std::string GPURenderer::PATH_TRACING_KERNEL_ID = "Path Tracing";
const std::string GPURenderer::RAY_VOLUME_STATE_SIZE_KERNEL_ID = "Ray Volume State Size";

// List of partials_options that will be specific to each kernel. We don't want these partials_options
	// to be synchronized between kernels
const std::unordered_set<std::string> GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED =
{
	GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
};

const std::unordered_map<std::string, std::string> GPURenderer::KERNEL_FUNCTION_NAMES = 
{
	{ CAMERA_RAYS_KERNEL_ID, "CameraRays" },
	{ PATH_TRACING_KERNEL_ID, "FullPathTracer" },
	{ RAY_VOLUME_STATE_SIZE_KERNEL_ID, "RayVolumeStateSize" },
};

const std::unordered_map<std::string, std::string> GPURenderer::KERNEL_FILES =
{
	{ CAMERA_RAYS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/CameraRays.h" },
	{ PATH_TRACING_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/FullPathTracer.h" },
	{ RAY_VOLUME_STATE_SIZE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/Utils/RayVolumeStateSize.h" },
};

const std::string GPURenderer::ALL_RENDER_PASSES_TIME_KEY = "FullFrameTime";
const std::string GPURenderer::FULL_FRAME_TIME_WITH_CPU_KEY = "FullFrameTimeWithCPU";
const std::string GPURenderer::DEBUG_KERNEL_TIME_KEY = "DebugKernelTime";

GPURenderer::GPURenderer(std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx)
{
	m_rng.m_state.seed = 42;

	// Creating buffers
	m_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_denoised_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_normals_AOV_interop_buffer = std::make_shared<OpenGLInteropBuffer<float3>>();
	m_denoiser_buffers.m_normals_AOV_no_interop_buffer = std::make_shared<OrochiBuffer<float3>>();
	m_denoiser_buffers.m_albedo_AOV_interop_buffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoiser_buffers.m_albedo_AOV_no_interop_buffer = std::make_shared<OrochiBuffer<ColorRGB32F>>();
	m_pixels_converged_sample_count_buffer = std::make_shared<OrochiBuffer<int>>();
	
	m_hiprt_orochi_ctx = hiprt_oro_ctx;	
	m_device_properties = m_hiprt_orochi_ctx->device_properties;

	setup_brdfs_data();
	setup_filter_functions();
	setup_kernels();

	m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY] = 0.0f;
	for (auto& id_to_pass : GPURenderer::KERNEL_FUNCTION_NAMES)
		m_render_pass_times[id_to_pass.first] = 0.0f;

	OROCHI_CHECK_ERROR(oroStreamCreate(&m_main_stream));

	// Buffer that keeps track of whether at least one ray is still alive or not
	m_status_buffers.still_one_ray_active_buffer.resize(1);
	m_status_buffers.still_one_ray_active_buffer.memset(1);
	m_status_buffers.pixels_converged_count_buffer.resize(1);

	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_start_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_stop_event));
}

void GPURenderer::setup_brdfs_data()
{
	init_sheen_ltc_texture();

	init_GGX_Ess_texture();
	init_glossy_dielectric_Ess_texture();
	init_GGX_glass_Ess_texture();
}

void GPURenderer::init_sheen_ltc_texture()
{
	// CUDA/HIP do not handle 3 channels textures so we're padding it to 4 channels
	std::vector<float> padded_ltc(32 * 32 * 4);

	for (int y = 0; y < 32; y++)
	{
		for (int x = 0; x < 32; x++)
		{
			int padded_index = (y * 32 + x) * 4;
			int non_padded_index = y * 32 + x;

			padded_ltc[padded_index + 0] = ltc_parameters_table_approximation[non_padded_index].x;
			padded_ltc[padded_index + 1] = ltc_parameters_table_approximation[non_padded_index].y;
			padded_ltc[padded_index + 2] = ltc_parameters_table_approximation[non_padded_index].z;
			padded_ltc[padded_index + 3] = 0.0f;
		}
	}

	Image32Bit sheen_ltc_params_image(padded_ltc.data(), 32, 32, 4);
	m_sheen_ltc_params = OrochiTexture(sheen_ltc_params_image, hipFilterModeLinear, hipAddressModeClamp);
}

void GPURenderer::init_GGX_Ess_texture(hipTextureFilterMode filtering_mode)
{
	Image32Bit GGXEss_image = Image32Bit::read_image_hdr(BRDFS_DATA_DIRECTORY "/GGX/" + GPUBakerConstants::get_GGX_conductor_Ess_filename(), 1, true);
	m_GGX_conductor_Ess = OrochiTexture(GGXEss_image, filtering_mode, hipAddressModeClamp);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::init_glossy_dielectric_Ess_texture(hipTextureFilterMode filtering_mode)
{
	synchronize_kernel();

	std::vector<Image32Bit> images(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_glossy_dielectric_Ess_filename();
		std::string filepath = BRDFS_DATA_DIRECTORY "/GlossyDielectrics/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_glossy_dielectric_Ess = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::init_GGX_glass_Ess_texture(hipTextureFilterMode filtering_mode)
{
	synchronize_kernel();

	std::vector<Image32Bit> images(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_Ess_filename();
		std::string filepath = BRDFS_DATA_DIRECTORY "/GGX/Glass/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_Ess_glass = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	for (int i = 0; i < GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_glass_inv_Ess_filename();
		std::string filepath = BRDFS_DATA_DIRECTORY "/GGX/Glass/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_Ess_glass_inverse = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	images.resize(GPUBakerConstants::GGX_THIN_GLASS_ESS_TEXTURE_SIZE_IOR);
	for (int i = 0; i < GPUBakerConstants::GGX_THIN_GLASS_ESS_TEXTURE_SIZE_IOR; i++)
	{
		std::string filename = std::to_string(i) + GPUBakerConstants::get_GGX_thin_glass_Ess_filename();
		std::string filepath = BRDFS_DATA_DIRECTORY "/GGX/Glass/" + filename;
		images[i] = Image32Bit::read_image_hdr(filepath, 1, true);
	}
	m_GGX_Ess_thin_glass = OrochiTexture3D(images, filtering_mode == hipFilterModeLinear ? ORO_TR_FILTER_MODE_LINEAR : ORO_TR_FILTER_MODE_POINT, ORO_TR_ADDRESS_MODE_CLAMP);

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::setup_nee_plus_plus_from_scene(const Scene& scene)
{
	m_nee_plus_plus.base_grid_min_point = scene.metadata.scene_bounding_box.mini;
	m_nee_plus_plus.base_grid_max_point = scene.metadata.scene_bounding_box.maxi;
}

void GPURenderer::reset_nee_plus_plus()
{
	m_render_data.nee_plus_plus.reset_visibility_map = true;
	m_render_data.nee_plus_plus.update_visibility_map = true;

	// Resetting the counters
	if (m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer() != nullptr)
	{
		m_nee_plus_plus.total_shadow_ray_queries.memset(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset(1);
	}
	m_nee_plus_plus.total_shadow_ray_queries_cpu = 1;
	m_nee_plus_plus.shadow_rays_actually_traced_cpu = 1;

	m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_START_TIMER;
}

NEEPlusPlusGPUData& GPURenderer::get_nee_plus_plus_data()
{
	return m_nee_plus_plus;
}

void GPURenderer::setup_filter_functions()
{
	// Function called on intersections for handling alpha testing
	hiprtFuncNameSet alpha_testing_func_set = { nullptr, "filter_function" };
	m_func_name_sets.push_back(alpha_testing_func_set);

	hiprtFuncDataSet func_data_set;
	hiprtFuncTable func_table;
	HIPRT_CHECK_ERROR(hiprtCreateFuncTable(m_hiprt_orochi_ctx->hiprt_ctx, 1, 1, func_table));
	HIPRT_CHECK_ERROR(hiprtSetFuncTable(m_hiprt_orochi_ctx->hiprt_ctx, func_table, 0, 0, func_data_set));

	m_render_data.hiprt_function_table = func_table;
}

void GPURenderer::setup_kernels()
{
	m_global_compiler_options = std::make_shared<GPUKernelCompilerOptions>();
	// Adding hardware acceleration by default if supported
	m_global_compiler_options->set_macro_value("__USE_HWI__", device_supports_hardware_acceleration() == HardwareAccelerationSupport::SUPPORTED);

	// Some default values are set for USE_SHARED_STACK_BVH_TRAVERSAL and SHARED_STACK_BVH_TRAVERSAL_SIZE
	// which I found work approximately well in terms of performance on various scenes (not perfect though and, on top of not 
	// being perfect, this was measured on a 7900XTX with hardware accelerated ray tracing so... your mileage in terms of what 
	// numbers are the best may vary.)
	
	// Configuring the kernels
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES.at(GPURenderer::CAMERA_RAYS_KERNEL_ID));
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTION_NAMES.at(GPURenderer::CAMERA_RAYS_KERNEL_ID));
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES.at(GPURenderer::PATH_TRACING_KERNEL_ID));
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTION_NAMES.at(GPURenderer::PATH_TRACING_KERNEL_ID));
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_restir_di_render_pass = ReSTIRDIRenderPass(this);
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
		// We only need to compile the ReSTIR DI render pass if ReSTIR DI is actually being used
		m_restir_di_render_pass.compile(m_hiprt_orochi_ctx, m_func_name_sets);

	// Configuring the kernel that will be used to retrieve the size of the RayVolumeState structure.
	// This size will be needed to resize the 'ray_volume_states' buffer in the GBuffer if the nested dielectrics
	// stack size changes
	//
	// We're compiling it serially so that we're sure that we can retrieve the RayVolumeState size on the GPU after the
	// GPURenderer is constructed
	m_ray_volume_state_byte_size_kernel.set_kernel_file_path(GPURenderer::KERNEL_FILES.at(GPURenderer::RAY_VOLUME_STATE_SIZE_KERNEL_ID));
	m_ray_volume_state_byte_size_kernel.set_kernel_function_name(GPURenderer::KERNEL_FUNCTION_NAMES.at(GPURenderer::RAY_VOLUME_STATE_SIZE_KERNEL_ID));
	m_ray_volume_state_byte_size_kernel.synchronize_options_with(*m_global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	ThreadManager::start_thread(ThreadManager::COMPILE_RAY_VOLUME_STATE_SIZE_KERNEL_KEY, ThreadFunctions::compile_kernel_silent, std::ref(m_ray_volume_state_byte_size_kernel), m_hiprt_orochi_ctx, std::ref(m_func_name_sets));

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE)
		m_nee_plus_plus.compile_finalize_accumulation_kernel(m_hiprt_orochi_ctx);

	// Compiling kernels
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID]), m_hiprt_orochi_ctx, std::ref(m_func_name_sets));
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID]), m_hiprt_orochi_ctx, std::ref(m_func_name_sets));
}

void GPURenderer::update(float delta_time)
{
	step_animations(delta_time);
	m_restir_di_render_pass.update();

	internal_update_clear_device_status_buffers();
	internal_update_prev_frame_g_buffer();
	internal_update_adaptive_sampling_buffers();
	internal_update_nee_plus_plus(delta_time);
	internal_update_global_stack_buffer();

	update_render_data();

	// Resetting this flag as this is a new frame
	m_render_data.render_settings.do_update_status_buffers = false;

	if (!m_render_data.render_settings.accumulate)
		m_render_data.render_settings.sample_number = 0;

	m_updated = true;
}

void GPURenderer::step_animations(float delta_time)
{
	m_envmap.update(this, delta_time);
	m_camera_animation.animation_step(this, delta_time);
}

void GPURenderer::download_status_buffers()
{
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.one_ray_active, m_status_buffers.still_one_ray_active_buffer.get_device_pointer(), sizeof(unsigned char), oroMemcpyDeviceToHost));
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.pixel_converged_count, m_status_buffers.pixels_converged_count_buffer.get_device_pointer(), sizeof(unsigned int), oroMemcpyDeviceToHost));
}

void GPURenderer::internal_update_clear_device_status_buffers()
{
	unsigned char false_data = false;
	unsigned int zero_data = 0;
	// Uploading false to reset the flag
	m_status_buffers.still_one_ray_active_buffer.upload_data(&false_data);
	// Resetting the counter of pixels converged to 0
	m_status_buffers.pixels_converged_count_buffer.upload_data(&zero_data);
}

void GPURenderer::internal_clear_m_status_buffers()
{
	m_status_buffers_values.one_ray_active = true;
	m_status_buffers_values.pixel_converged_count = 0;
}

void GPURenderer::internal_update_prev_frame_g_buffer()
{
	if (m_render_data.render_settings.use_prev_frame_g_buffer(this))
	{
		// If at least one buffer has a size of 0, we assume that this means that the whole G-buffer is deallocated
		// and so we're going to have to reallocate it
		bool prev_frame_g_buffer_needs_resize = m_g_buffer_prev_frame.first_hit_prim_index.get_element_count() == 0;

		if (prev_frame_g_buffer_needs_resize)
		{
			m_g_buffer_prev_frame.resize(m_render_resolution.x * m_render_resolution.y, get_ray_volume_state_byte_size());
			m_render_data_buffers_invalidated = true;
		}
	}
	else
	{
		// If we're not using the G-buffer, indicating that in use_last_frame_g_buffer so that the shader doesn't
		// try to use it

		if (m_g_buffer_prev_frame.first_hit_prim_index.get_element_count() > 0)
		{
			// If the buffers aren't freed already
			m_g_buffer_prev_frame.free();
			m_render_data_buffers_invalidated = true;
		}
	}
}

void GPURenderer::internal_update_adaptive_sampling_buffers()
{
	bool buffers_needed = m_render_data.render_settings.has_access_to_adaptive_sampling_buffers();

	if (buffers_needed)
	{
		bool pixels_squared_luminance_needs_resize = m_pixels_squared_luminance_buffer.get_element_count() == 0;
		bool pixels_sample_count_needs_resize = m_pixels_sample_count_buffer.get_element_count() == 0;
		bool pixels_converged_sample_count_needs_resize = m_pixels_converged_sample_count_buffer->get_element_count() == 0;

		if (pixels_squared_luminance_needs_resize || pixels_sample_count_needs_resize || pixels_converged_sample_count_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			m_render_data_buffers_invalidated = true;

		if (pixels_squared_luminance_needs_resize)
			// Only allocating if it isn't already
			m_pixels_squared_luminance_buffer.resize(m_render_resolution.x * m_render_resolution.y);

		if (pixels_sample_count_needs_resize)
			// Only allocating if it isn't already
			m_pixels_sample_count_buffer.resize(m_render_resolution.x * m_render_resolution.y);

		if (pixels_converged_sample_count_needs_resize)
			m_pixels_converged_sample_count_buffer->resize(m_render_resolution.x * m_render_resolution.y);

	}
	else
	{
		if (m_pixels_squared_luminance_buffer.get_element_count() > 0 || m_pixels_sample_count_buffer.get_element_count() > 0 || m_pixels_converged_sample_count_buffer->get_element_count() > 0)
			m_render_data_buffers_invalidated = true;

		m_pixels_squared_luminance_buffer.free();
		m_pixels_sample_count_buffer.free();
		m_pixels_converged_sample_count_buffer->free();
	}
}

void GPURenderer::internal_update_nee_plus_plus(float delta_time)
{
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_FALSE)
	{
		// Not using NEE++, we just need to free the buffers if they weren't already

		if (m_nee_plus_plus.packed_buffer.get_element_count() != 0)
		{
			m_nee_plus_plus.packed_buffer.free();
			m_nee_plus_plus.total_shadow_ray_queries.free();
			m_nee_plus_plus.shadow_rays_actually_traced.free();

			m_render_data_buffers_invalidated = true;
		}

		return;
	}

	float3 min_grid_extent_with_envmap, max_grid_extent_with_envmap;
	m_nee_plus_plus.get_grid_extents(m_nee_plus_plus.grid_dimensions_no_envmap, min_grid_extent_with_envmap, max_grid_extent_with_envmap);

	// Adding (2, 2, 2) for envmap NEE++
	m_render_data.nee_plus_plus.grid_dimensions = m_nee_plus_plus.grid_dimensions_no_envmap + make_int3(2, 2, 2);
	m_render_data.nee_plus_plus.grid_min_point = min_grid_extent_with_envmap;
	m_render_data.nee_plus_plus.grid_max_point = max_grid_extent_with_envmap;

	// Allocating / deallocating buffers
	unsigned int matrix_element_count = m_nee_plus_plus.get_visibility_matrix_element_count(m_render_data.nee_plus_plus.grid_dimensions);
	if (m_nee_plus_plus.packed_buffer.get_element_count() != matrix_element_count)
	{
		m_nee_plus_plus.packed_buffer.resize(matrix_element_count);
		m_nee_plus_plus.shadow_rays_actually_traced.resize(1);
		m_nee_plus_plus.total_shadow_ray_queries.resize(1);

		m_render_data_buffers_invalidated = true;
	}

	// Clearing the visibility map if this has been asked by the user
	if (m_render_data.nee_plus_plus.reset_visibility_map)
	{
		// Clearing the visibility map by memseting everything to 0
		m_nee_plus_plus.packed_buffer.memset(0);
		m_nee_plus_plus.total_shadow_ray_queries.memset(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset(1);
	}

	if (m_render_data.render_settings.sample_number > m_nee_plus_plus.stop_update_samples)
		// Past a certain number of samples, there isn't really a point to keep updating, the visibility map
		// is probably converged enough that it doesn't make a difference anymore
		m_render_data.nee_plus_plus.update_visibility_map = false;

	m_nee_plus_plus.milliseconds_before_finalizing_accumulation -= delta_time;
	m_nee_plus_plus.milliseconds_before_finalizing_accumulation = hippt::max(0.0f, m_nee_plus_plus.milliseconds_before_finalizing_accumulation); // Clamping for nice display in ImGui (0.0f instead of negative values)
	if (m_nee_plus_plus.milliseconds_before_finalizing_accumulation <= 0.0f && m_render_data.nee_plus_plus.packed_buffers != nullptr)
	{
		m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_TIMER;

		// Because the visibility map data is packed, we can't just use a memcpy() to copy from the accumulation
		// buffers to the visibilit map, we have to use a kernel that the does unpacking-copy
		void* launch_args[] = { &m_render_data.nee_plus_plus };
		m_nee_plus_plus.finalize_accumulation_kernel.launch_asynchronous(256, 1, matrix_element_count, 1, launch_args, m_main_stream);
	}
	
	m_nee_plus_plus.statistics_refresh_timer -= delta_time;
	if (m_nee_plus_plus.statistics_refresh_timer <= 0.0f && m_render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics)
	{
		m_nee_plus_plus.statistics_refresh_timer = NEEPlusPlusGPUData::STATISTICS_REFRESH_TIMER;

		OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.total_shadow_ray_queries_cpu, m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
		OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.shadow_rays_actually_traced_cpu, m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
	}
}

void GPURenderer::internal_update_global_stack_buffer()
{
	if (needs_global_bvh_stack_buffer())
	{
		bool buffer_needs_update = false;
		// Buffer isn't allocated
		buffer_needs_update |= m_render_data.global_traversal_stack_buffer.stackData == nullptr;
		// Buffer is allocated but the stack size has been changed (through ImGui probably)
		buffer_needs_update |= m_render_data.global_traversal_stack_buffer_size != m_render_data.global_traversal_stack_buffer.stackSize;

		if (buffer_needs_update)
			recreate_global_bvh_stack_buffer();
	}
	else
	{
		if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
		{
			// Freeing if the buffer already exists
			HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));
			m_render_data.global_traversal_stack_buffer.stackData = nullptr;
		}
	}
}

bool GPURenderer::needs_global_bvh_stack_buffer()
{
	for (const auto& name_to_kernel : m_kernels)
	{
		bool global_stack_buffer_needed = false;
		global_stack_buffer_needed |= name_to_kernel.second.get_kernel_options().get_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL) == KERNEL_OPTION_TRUE;

		if (global_stack_buffer_needed)
			return true;
	}

	return false;
}

void GPURenderer::recreate_global_bvh_stack_buffer()
{
	int nbBlocksX = std::ceil(m_render_resolution.x / (float)KernelBlockWidthHeight) * KernelBlockWidthHeight;
	int nbBlocksY = std::ceil(m_render_resolution.y / (float)KernelBlockWidthHeight) * KernelBlockWidthHeight;

	// Resizing the global stack buffer for BVH traversal
	hiprtGlobalStackBufferInput stackBufferInput
	{
		hiprtStackTypeGlobal,
		hiprtStackEntryTypeInteger,
		static_cast<uint32_t>(m_render_data.global_traversal_stack_buffer_size),
		static_cast<uint32_t>(nbBlocksX * nbBlocksY)
	};

	if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
		// Freeing if the buffer already exists
		HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));

	HIPRT_CHECK_ERROR(hiprtCreateGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, stackBufferInput, m_render_data.global_traversal_stack_buffer));
}

void GPURenderer::render()
{
	if (!m_updated)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "render() was called on the GPURenderer without update() being called.");
		Utils::debugbreak();

		return;
	}

	// Resetting the update state since we're now rendering a new frame
	m_updated = false;

	// Making sure kernels are compiled
	ThreadManager::join_threads(ThreadManager::COMPILE_KERNELS_THREAD_KEY);

	map_buffers_for_render();
	
	if (m_debug_trace_kernel.has_been_compiled())
		render_debug_kernel();
	else
		render_path_tracing();
}

void GPURenderer::render_debug_kernel()
{
	m_frame_rendered = false;

	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_start_event, m_main_stream));

	// Updating the previous and current camera
	m_render_data.current_camera = m_camera.to_hiprt();
	m_render_data.prev_camera = m_previous_frame_camera.to_hiprt();

	launch_debug_kernel();

	// Recording GPU frame time stop timestamp and computing the frame time
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, m_main_stream));
	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, [](void* payload) {
		*reinterpret_cast<bool*>(payload) = true;
	}, &m_frame_rendered));
}

void GPURenderer::render_path_tracing()
{
	m_frame_rendered = false;

	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_start_event, m_main_stream));

	for (int i = 1; i <= m_render_data.render_settings.samples_per_frame; i++)
	{
		// Updating the previous and current camera
		m_render_data.current_camera = m_camera.to_hiprt();
		m_render_data.prev_camera = m_previous_frame_camera.to_hiprt();

		if (i == m_render_data.render_settings.samples_per_frame)
			// Last sample of the frame so we are going to enable the update 
			// of the status buffers (number of pixels converged, how many rays still
			// active, ...)
			m_render_data.render_settings.do_update_status_buffers = true;
		
		launch_camera_rays();
		launch_ReSTIR_DI();
		launch_path_tracing();

		m_render_data.render_settings.sample_number++;
		m_render_data.render_settings.denoiser_AOV_accumulation_counter++;

		// We only reset once so after rendering a frame, we're sure that we don't need to reset anymore 
		// so we're setting the flag to false (it will be set to true again if we need to reset the render
		// again)
		m_render_data.render_settings.need_to_reset = false;
		m_render_data.nee_plus_plus.reset_visibility_map = false;
		// If we had requested a temporal buffers clear, this has be done by this frame so we can
		// now reset the flag
		m_render_data.render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = false;

		// Saving the current frame camera to be the previous camera of the next frame
		m_previous_frame_camera = m_camera;
	}

	// Recording GPU frame time stop timestamp and computing the frame time
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, m_main_stream));
	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, [](void* payload){
		*reinterpret_cast<bool*>(payload) = true;
	}, &m_frame_rendered));

	m_was_last_frame_low_resolution = m_render_data.render_settings.do_render_low_resolution();
	// We just rendered a new frame so we're setting this flag to true
	// such that the animated components of the scene are not allowed to step
	// their animations until the render window signals the renderer the the
	// frame has been fully rendered and thus that the animations can step forward
	m_animation_state.can_step_animation = false;
}

void GPURenderer::launch_nee_plus_plus_caching_prepass()
{
	unsigned int caching_sample_count = 1024;
	void* launch_args[] = { &m_render_data, &caching_sample_count, &m_render_resolution };

	m_render_data.random_seed = m_rng.xorshift32();
	m_kernels[GPURenderer::NEE_PLUS_PLUS_CACHING_PREPASS_ID].launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::launch_camera_rays()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	m_render_data.random_seed = m_rng.xorshift32();
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::launch_ReSTIR_DI()
{
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
		// Only launching if ReSTIR DI is enabled
		m_restir_di_render_pass.launch();
}

void GPURenderer::launch_path_tracing()
{
	void* launch_args[] = { &m_render_data };

	m_render_data.random_seed = m_rng.xorshift32();
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::launch_debug_kernel()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	m_render_data.random_seed = m_rng.xorshift32();
	m_debug_trace_kernel.launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::synchronize_kernel()
{
	if (m_main_stream == nullptr)
		return;

	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_main_stream));
}

bool GPURenderer::frame_render_done()
{
	return m_frame_rendered;
}

bool GPURenderer::was_last_frame_low_resolution()
{
	return m_was_last_frame_low_resolution;
}

void GPURenderer::resize(int new_width, int new_height, bool also_resize_interop)
{
	// Needed so that this function can eventually be called from another thread
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

	m_render_resolution = make_int2(new_width, new_height);

	synchronize_kernel();
	unmap_buffers();

	if (also_resize_interop)
		resize_interop_buffers(new_width, new_height);

	m_g_buffer.resize(new_width * new_height, get_ray_volume_state_byte_size());

	if (m_render_data.render_settings.use_prev_frame_g_buffer(this))
		m_g_buffer_prev_frame.resize(new_width * new_height, get_ray_volume_state_byte_size());

	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		m_pixels_squared_luminance_buffer.resize(new_width * new_height);
		m_pixels_sample_count_buffer.resize(new_width * new_height);
	}

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
		m_restir_di_render_pass.resize(new_width, new_height);

	m_pixel_active.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.set_aspect(new_aspect);

	if (needs_global_bvh_stack_buffer())
		recreate_global_bvh_stack_buffer();

	m_render_data.render_settings.render_resolution = m_render_resolution;
	m_render_data_buffers_invalidated = true;
}

void GPURenderer::resize_interop_buffers(int new_width, int new_height)
{
	m_framebuffer->resize(new_width * new_height);
	m_denoiser_buffers.m_denoised_framebuffer->resize(new_width * new_height);
	m_denoiser_buffers.resize_normals_buffer(new_width * new_height);
	m_denoiser_buffers.resize_albedo_buffer(new_width * new_height);

	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		m_pixels_converged_sample_count_buffer->resize(new_width * new_height);
}

void GPURenderer::map_buffers_for_render()
{
	m_render_data.buffers.pixels = m_framebuffer->map_no_error();
	m_render_data.aux_buffers.denoiser_normals = m_denoiser_buffers.map_normals_buffer();
	m_render_data.aux_buffers.denoiser_albedo = m_denoiser_buffers.map_albedo_buffer();
	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		m_render_data.aux_buffers.pixel_converged_sample_count = m_pixels_converged_sample_count_buffer->get_device_pointer();
}

void GPURenderer::unmap_buffers()
{
	m_framebuffer->unmap();
	m_denoiser_buffers.unmap_normals_buffer();
	m_denoiser_buffers.unmap_albedo_buffer();
	//m_pixels_converged_sample_count_buffer->unmap();
}


void GPURenderer::set_use_denoiser_AOVs_interop_buffers(bool use_interop) { m_denoiser_buffers.set_use_interop_AOV_buffers(this, use_interop); }

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_color_interop_framebuffer() { return m_framebuffer; }
std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoised_interop_framebuffer() { return m_denoiser_buffers.m_denoised_framebuffer;}
std::shared_ptr<OpenGLInteropBuffer<float3>> GPURenderer::get_denoiser_normals_AOV_interop_buffer() 
{
	if (!m_denoiser_buffers.use_interop_AOVs)
		// No using the interop buffers so let's not return a buffer that cannot be used
		return nullptr;

	return m_denoiser_buffers.m_normals_AOV_interop_buffer; 
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoiser_albedo_AOV_interop_buffer() 
{ 
	if (!m_denoiser_buffers.use_interop_AOVs)
		// No using the interop buffers so let's not return a buffer that cannot be used
		return nullptr;

	return m_denoiser_buffers.m_albedo_AOV_interop_buffer; 
}

std::shared_ptr<OrochiBuffer<float3>> GPURenderer::get_denoiser_normals_AOV_no_interop_buffer() { return m_denoiser_buffers.m_normals_AOV_no_interop_buffer; }
std::shared_ptr<OrochiBuffer<ColorRGB32F>> GPURenderer::get_denoiser_albedo_AOV_no_interop_buffer() { return m_denoiser_buffers.m_albedo_AOV_no_interop_buffer; }

std::shared_ptr<OrochiBuffer<int>>& GPURenderer::get_pixels_converged_sample_count_buffer() { return m_pixels_converged_sample_count_buffer; }
const StatusBuffersValues& GPURenderer::get_status_buffer_values() const { return m_status_buffers_values; }

HIPRTRenderSettings& GPURenderer::get_render_settings() { return m_render_data.render_settings; }
WorldSettings& GPURenderer::get_world_settings() { return m_render_data.world_settings; }
HIPRTRenderData& GPURenderer::get_render_data() { return m_render_data; }
HIPRTScene& GPURenderer::get_hiprt_scene() { return m_hiprt_scene; }
std::shared_ptr<HIPRTOrochiCtx> GPURenderer::get_hiprt_orochi_ctx() { return m_hiprt_orochi_ctx; }

void GPURenderer::invalidate_render_data_buffers() { m_render_data_buffers_invalidated = true; }
oroDeviceProp GPURenderer::get_device_properties() { return m_device_properties;}

std::string getDeviceName(oroCtx m_ctxt, oroDevice m_device)
{
	oroDeviceProp prop;
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_ctxt));
	OROCHI_CHECK_ERROR(oroGetDeviceProperties(&prop, m_device));
	return std::string(prop.name);
}

std::string getGcnArchName(oroCtx m_ctxt, oroDevice m_device)
{
	oroDeviceProp prop;
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_ctxt));
	OROCHI_CHECK_ERROR(oroGetDeviceProperties(&prop, m_device));
	return std::string(prop.gcnArchName);
}

uint32_t getGcnArchNumber(oroCtx m_ctxt, oroDevice m_device)
{
	oroDeviceProp prop;
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_ctxt));
	OROCHI_CHECK_ERROR(oroGetDeviceProperties(&prop, m_device));
	return prop.gcnArch;
}

bool enableHwi(oroCtx m_ctxt, oroDevice m_device)
{
	std::string	   deviceName = getDeviceName(m_ctxt, m_device);
	const uint32_t archNumber = getGcnArchNumber(m_ctxt, m_device);
	return (archNumber >= 1030 && deviceName.find("NVIDIA") == std::string::npos);
}

HardwareAccelerationSupport GPURenderer::device_supports_hardware_acceleration()
{
	bool enabled = reinterpret_cast<hiprt::Context*>(m_hiprt_orochi_ctx->hiprt_ctx)->enableHwi();
	if (enabled)
		return HardwareAccelerationSupport::SUPPORTED;
	else
	{
		if (std::string(m_device_properties.name).find("NVIDIA") != std::string::npos)
		{
			// Not supported on NVIDIA
			return HardwareAccelerationSupport::NVIDIA_UNSUPPORTED;
		}
		else
		{
			// Not NVIDIA but hardware acceleration not supported, assuming too old AMD
			return HardwareAccelerationSupport::AMD_UNSUPPORTED;
		}
	}
}

std::shared_ptr<GPUKernelCompilerOptions> GPURenderer::get_global_compiler_options()
{
	return m_global_compiler_options;
}

// Variables used to give the priority to the main thread when compiling shaders
extern bool g_main_thread_compiling;
extern std::condition_variable g_condition_for_compilation;

void GPURenderer::recompile_kernels(bool use_cache)
{
	synchronize_kernel();

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Recompiling kernels...");

	// Notifying all threads that may be compiling that the main thread wants to
	// compile. This will block threads other than the main thread from compiling
	// and thus give the priority to the main thread
	take_kernel_compilation_priority();

	for (auto& name_to_kenel : m_kernels)
		name_to_kenel.second.compile_silent(m_hiprt_orochi_ctx, m_func_name_sets, use_cache);

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
		// We only need to compile the ReSTIR DI render pass if ReSTIR DI is actually being used
		m_restir_di_render_pass.recompile(m_hiprt_orochi_ctx, m_func_name_sets, true, use_cache);

	m_ray_volume_state_byte_size_kernel.compile_silent(m_hiprt_orochi_ctx, m_func_name_sets, use_cache);

	// The main thread is done with the compilation, we can release the other threads
	// so that they can continue compiling (background compilation of shaders most likely)
	release_kernel_compilation_priority();
}

// Variables used to give the priority to the main thread when compiling shaders
extern std::thread::id g_priority_thread_id;
extern bool g_main_thread_compiling;
extern std::condition_variable g_condition_for_compilation;
void GPURenderer::take_kernel_compilation_priority()
{
	// Notifying all threads that may be compiling that the main thread wants to
	// compile. This will block threads other than the main thread from compiling
	// and thus give the priority to the main thread
	g_main_thread_compiling = true;
	g_condition_for_compilation.notify_all();
	g_priority_thread_id = std::this_thread::get_id();
}

void GPURenderer::release_kernel_compilation_priority()
{
	// The main thread is done with the compilation, we can release the other threads
	// so that they can continue compiling (background compilation of shaders most likely)
	g_main_thread_compiling = false;
	g_condition_for_compilation.notify_all();
}

void GPURenderer::precompile_kernels()
{
	return;

	g_imgui_logger.add_line_with_name(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, ImGuiLogger::BACKGROUND_KERNEL_PARSING_LINE_NAME, "Parsing kernel permutations in the background... [%d / %d]", 0, 1);
	g_imgui_logger.add_line_with_name(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, ImGuiLogger::BACKGROUND_KERNEL_COMPILATION_LINE_NAME, "Compiling kernel permutations in the background... [%d / %d]", 0, 1);

	// Launching all the threads actually takes some time
	// so we're launching threads from a thread :D
	// 
	// We're not going to join the thread started right below
	// so we can use a const char* for the key, we don't a constant
	// defined in ThreadManager. Quick and dirty.
	ThreadManager::start_thread(ThreadManager::GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY, [this]() {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		precompile_direct_light_sampling_kernels();
		precompile_ReSTIR_DI_kernels();
	});

	ThreadManager::detach_threads(ThreadManager::GPU_RENDERER_PRECOMPILE_KERNELS_THREAD_KEY);
}

extern bool g_background_shader_compilation_enabled;
void GPURenderer::stop_background_shader_compilation()
{
	g_background_shader_compilation_enabled = false;
	g_condition_for_compilation.notify_all();
}

void GPURenderer::resume_background_shader_compilation()
{
	g_background_shader_compilation_enabled = true;
	g_condition_for_compilation.notify_all();
}

void GPURenderer::precompile_direct_light_sampling_kernels()
{
	for (int init_target_function_vis = 0; init_target_function_vis <= 1; init_target_function_vis++)
	{
		for (int use_envmap_mis = 0; use_envmap_mis <= 1; use_envmap_mis++)
		{
			for (int envmap_sampling_strategy = ESS_NO_SAMPLING; envmap_sampling_strategy < ESS_ALIAS_TABLE; envmap_sampling_strategy++)
			{
				for (int direct_light_sampling_strategy = LSS_NO_DIRECT_LIGHT_SAMPLING; direct_light_sampling_strategy <= LSS_RESTIR_DI - 1; direct_light_sampling_strategy++)
				{
					// Starting from what the renderer is currently using to ease our life a little
					// (partials_options like USE_HWI, BVH_TRAVERSAL_STACK_SIZE, ... would have to be copied
					// manually otherwise so just copying everything here is handy)
					GPUKernelCompilerOptions partials_options;
					// Clearing the default state of the partials_options added by the constructor
					partials_options.clear();
					partials_options.set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, direct_light_sampling_strategy);
					partials_options.set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, envmap_sampling_strategy);
					partials_options.set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS, use_envmap_mis);

					// Recompiling all the kernels with the new options
					for (const std::string& kernel_id : get_all_kernel_ids())
						precompile_kernel(kernel_id, partials_options);
					m_restir_di_render_pass.precompile_kernels(partials_options, m_hiprt_orochi_ctx, m_func_name_sets);

					if (direct_light_sampling_strategy == LSS_RIS_BSDF_AND_LIGHT)
					{
						// Additional compilation for RIS with the visibility in the target function
						// for the value we haven't compiled yet
						partials_options.set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, 1 - m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION));

						// Recompiling all the kernels with the new options
						for (const std::string& kernel_id : get_all_kernel_ids())
							precompile_kernel(kernel_id, partials_options);
						m_restir_di_render_pass.precompile_kernels(partials_options, m_hiprt_orochi_ctx, m_func_name_sets);
					}
				}
			}
		}
	}
}

void GPURenderer::precompile_ReSTIR_DI_kernels()
{
	for (int init_target_function_vis = 0; init_target_function_vis <= 1; init_target_function_vis++)
	{
		for (int spatial_target_function_vis = 0; spatial_target_function_vis <= 1; spatial_target_function_vis++)
		{
			for (int do_light_presampling = 0; do_light_presampling <= 1; do_light_presampling++)
			{
				for (int visibility_bias_correction = 0; visibility_bias_correction <= 1; visibility_bias_correction++)
				{
					for (int do_visibility_reuse = 0; do_visibility_reuse <= 1; do_visibility_reuse++)
					{
						for (int bias_correction_weight = RESTIR_DI_BIAS_CORRECTION_1_OVER_M; bias_correction_weight <= RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE; bias_correction_weight++)
						{
							// Starting from what the renderer is currently using to ease our life a little
							// (partials_options like USE_HWI, BVH_TRAVERSAL_STACK_SIZE, ... would have to be copied
							// manually otherwise so just copying everything here is handy)
							GPUKernelCompilerOptions partials_options;
							// Clearing the default state of the partials_options added by the constructor
							partials_options.clear();
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY, init_target_function_vis);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, spatial_target_function_vis);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, do_visibility_reuse);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY, visibility_bias_correction);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS, bias_correction_weight);
							partials_options.set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHTS_PRESAMPLING, do_light_presampling);
							partials_options.set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_RESTIR_DI);

							// Recompiling all the kernels with the new options
							for (const std::string& kernel_id : get_all_kernel_ids())
								precompile_kernel(GPURenderer::NEE_PLUS_PLUS_CACHING_PREPASS_ID, partials_options);
							// Recompiling the ReSTIR DI render pass
							m_restir_di_render_pass.precompile_kernels(partials_options, m_hiprt_orochi_ctx, m_func_name_sets);
						}
					}
				}
			}
		}
	}
}

void GPURenderer::precompile_kernel(const std::string& id, GPUKernelCompilerOptions partial_options)
{
	GPUKernelCompilerOptions options = m_kernels[id].get_kernel_options();
	partial_options.apply_onto(options);

	ThreadManager::start_thread(ThreadManager::RENDERER_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		GPURenderer::KERNEL_FUNCTION_NAMES.at(id),
		GPURenderer::KERNEL_FILES.at(id),
		options, m_hiprt_orochi_ctx, std::ref(m_func_name_sets));

	ThreadManager::detach_threads(ThreadManager::RENDERER_PRECOMPILE_KERNELS);
}

std::map<std::string, GPUKernel*> GPURenderer::get_kernels()
{
	std::map<std::string, GPUKernel*> kernels;

	for (auto& pair : m_kernels)
		kernels[pair.first] = &pair.second;

	for (auto& pair : m_restir_di_render_pass.m_kernels)
		kernels[pair.first] = &pair.second;

	return kernels;
}

std::vector<std::string> GPURenderer::get_all_kernel_ids()
{
	std::vector<std::string> all_ids;
	// Size - 1 because we're going to excluded the ray volume state kernel
	all_ids.reserve(GPURenderer::KERNEL_FUNCTION_NAMES.size() - 1);

	for (const auto& kernel_function_name : GPURenderer::KERNEL_FUNCTION_NAMES)
	{
		const std::string& kernel_id = kernel_function_name.first;
		if (kernel_id == GPURenderer::RAY_VOLUME_STATE_SIZE_KERNEL_ID)
			// Ignoring this guy because he is special and we don't want it
			// in the render pass times
			continue;

		all_ids.push_back(kernel_id);
	}

	return all_ids;
}

void GPURenderer::set_debug_trace_kernel(const std::string& kernel_name, GPUKernelCompilerOptions options)
{
	if (kernel_name == "")
		// Clearing the debug kernel
		m_debug_trace_kernel = GPUKernel();
	else
	{
		m_debug_trace_kernel = GPUKernel(DEVICE_KERNELS_DIRECTORY "/" + kernel_name + ".h", kernel_name);

		// Setting all the custom options
		m_debug_trace_kernel.get_kernel_options() = options;
		m_debug_trace_kernel.compile(m_hiprt_orochi_ctx);
	}
}

bool GPURenderer::is_using_debug_kernel()
{
	return m_debug_trace_kernel.has_been_compiled();
}

oroStream_t GPURenderer::get_main_stream()
{
	return m_main_stream;
}

void GPURenderer::compute_render_pass_times()
{
	// Registering the render times of all the kernels by iterating over all the kernels
	for (const std::string& kernel_id : get_all_kernel_ids())
		m_render_pass_times[kernel_id] = m_kernels[kernel_id].get_last_execution_time();

	m_restir_di_render_pass.compute_render_times(m_render_pass_times);
	if (m_debug_trace_kernel.has_been_compiled())
		// If the debug kernel is being used... read its execution time
		// Note that we check for 'has_been_compiled()' because if the debug kernel isn't in use,
		// then the kernel (m_debug_trace_kernel) is empty, and if it's empty, then it hasn't
		// been compiled yet
		m_render_pass_times[GPURenderer::DEBUG_KERNEL_TIME_KEY] = m_debug_trace_kernel.get_last_execution_time();

	// The total frame time is the sum of every passes
	float sum = 0.0f;
	for (auto pair : m_render_pass_times)
	{
		if (pair.first == GPURenderer::ALL_RENDER_PASSES_TIME_KEY)
			continue;

		sum += pair.second;
	}

	m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY] = sum;
}

std::unordered_map<std::string, float>& GPURenderer::get_render_pass_times()
{
	return m_render_pass_times;
}

float GPURenderer::get_last_frame_time()
{
	return m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY];
}

void GPURenderer::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	compute_render_pass_times();

	// Also adding the times of the various passes
	for (const std::string& kernel_id : get_all_kernel_ids())
		perf_metrics->add_value(kernel_id, m_render_pass_times[kernel_id]);
	m_restir_di_render_pass.update_perf_metrics(perf_metrics);
	perf_metrics->add_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY, m_render_pass_times[GPURenderer::ALL_RENDER_PASSES_TIME_KEY]);

	if (m_debug_trace_kernel.has_been_compiled())
		// Adding the time for the debug kernel if it is in use
		perf_metrics->add_value(GPURenderer::DEBUG_KERNEL_TIME_KEY, m_render_pass_times[GPURenderer::DEBUG_KERNEL_TIME_KEY]);
}

void GPURenderer::reset(std::shared_ptr<ApplicationSettings> application_settings)
{
	if (m_render_data.render_settings.accumulate)
	{
		// Only resetting the seed for deterministic rendering if we're accumulating.
		// If we're not accumulating, we want each frame of the render to be different
		// so we don't get into that if block and we don't reset the seed
		m_rng.m_state.seed = 42;

		m_restir_di_render_pass.reset();
	
		if (application_settings->auto_sample_per_frame)
			m_render_data.render_settings.samples_per_frame = 1;
	}

	m_render_data.render_settings.denoiser_AOV_accumulation_counter = 0;
	m_render_data.render_settings.sample_number = 0;
	m_render_data.render_settings.need_to_reset = true;

	reset_nee_plus_plus();

	internal_clear_m_status_buffers();
}

Xorshift32Generator& GPURenderer::rng()
{
	return m_rng;
}

void GPURenderer::update_render_data()
{
	// Always updating the random seed
	m_render_data.random_seed = m_rng.xorshift32();

	if (m_render_data_buffers_invalidated)
	{
		m_render_data.GPU_BVH = m_hiprt_scene.geometry.m_geometry;

		m_render_data.buffers.triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.geometry.m_mesh.triangleIndices);
		m_render_data.buffers.vertices_positions = reinterpret_cast<float3*>(m_hiprt_scene.geometry.m_mesh.vertices);
		m_render_data.buffers.has_vertex_normals = reinterpret_cast<unsigned char*>(m_hiprt_scene.has_vertex_normals.get_device_pointer());
		m_render_data.buffers.vertex_normals = reinterpret_cast<float3*>(m_hiprt_scene.vertex_normals.get_device_pointer());
		m_render_data.buffers.material_indices = reinterpret_cast<int*>(m_hiprt_scene.material_indices.get_device_pointer());
		m_render_data.buffers.materials_buffer = m_hiprt_scene.materials_buffer.get_device_pointer();
		m_render_data.buffers.material_opaque = m_hiprt_scene.material_opaque.get_device_pointer();
		m_render_data.buffers.emissive_triangles_count = m_hiprt_scene.emissive_triangles_count;
		m_render_data.buffers.emissive_triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_indices.get_device_pointer());

		m_render_data.bsdfs_data.sheen_ltc_parameters_texture = m_sheen_ltc_params.get_device_texture();
		m_render_data.bsdfs_data.GGX_conductor_Ess = m_GGX_conductor_Ess.get_device_texture();
		m_render_data.bsdfs_data.glossy_dielectric_Ess = m_glossy_dielectric_Ess.get_device_texture();
		m_render_data.bsdfs_data.GGX_Ess_glass = m_GGX_Ess_glass.get_device_texture();
		m_render_data.bsdfs_data.GGX_Ess_glass_inverse = m_GGX_Ess_glass_inverse.get_device_texture();
		m_render_data.bsdfs_data.GGX_Ess_thin_glass = m_GGX_Ess_thin_glass.get_device_texture();

		m_render_data.buffers.material_textures = reinterpret_cast<oroTextureObject_t*>(m_hiprt_scene.gpu_materials_textures.get_device_pointer());
		m_render_data.buffers.texcoords = reinterpret_cast<float2*>(m_hiprt_scene.texcoords_buffer.get_device_pointer());

		m_render_data.g_buffer = m_g_buffer.get_device_g_buffer();

		if (m_render_data.render_settings.use_prev_frame_g_buffer(this))
			// Only setting the pointers of the buffers if we're actually using the g-buffer of the previous frame
			m_render_data.g_buffer_prev_frame = m_g_buffer_prev_frame.get_device_g_buffer();
		else
		{
			m_render_data.g_buffer_prev_frame.materials = nullptr;
			m_render_data.g_buffer_prev_frame.geometric_normals = nullptr;
			m_render_data.g_buffer_prev_frame.shading_normals = nullptr;
			m_render_data.g_buffer_prev_frame.primary_hit_position = nullptr;
		}

		if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		{
			m_render_data.aux_buffers.pixel_sample_count = m_pixels_sample_count_buffer.get_device_pointer();
			m_render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance_buffer.get_device_pointer();
		}

		m_render_data.aux_buffers.pixel_active = m_pixel_active.get_device_pointer();
		m_render_data.aux_buffers.still_one_ray_active = m_status_buffers.still_one_ray_active_buffer.get_device_pointer();
		m_render_data.aux_buffers.stop_noise_threshold_converged_count = reinterpret_cast<AtomicType<unsigned int>*>(m_status_buffers.pixels_converged_count_buffer.get_device_pointer());

		m_restir_di_render_pass.update_render_data();

		m_render_data.nee_plus_plus.packed_buffers = reinterpret_cast<AtomicType<unsigned int>*>(m_nee_plus_plus.packed_buffer.get_device_pointer());
		m_render_data.nee_plus_plus.shadow_rays_actually_traced = reinterpret_cast<AtomicType<unsigned int>*>(m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer());
		m_render_data.nee_plus_plus.total_shadow_ray_queries = reinterpret_cast<AtomicType<unsigned int>*>(m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer());

		m_render_data_buffers_invalidated = false;
	}
}

void GPURenderer::set_hiprt_scene_from_scene(const Scene& scene)
{
	if (scene.triangle_indices.size() == 0)
		// Empty scene, nothing todo
		return;

	m_hiprt_scene.geometry.upload_indices(scene.triangle_indices);
	m_hiprt_scene.geometry.upload_vertices(scene.vertices_positions);
	m_hiprt_scene.geometry.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	rebuild_renderer_bvh(hiprtBuildFlagBitPreferHighQualityBuild, true);

	m_hiprt_scene.has_vertex_normals.resize(scene.has_vertex_normals.size());
	m_hiprt_scene.has_vertex_normals.upload_data(scene.has_vertex_normals.data());

	m_hiprt_scene.vertex_normals.resize(scene.vertex_normals.size());
	m_hiprt_scene.vertex_normals.upload_data(scene.vertex_normals.data());

	m_hiprt_scene.material_indices.resize(scene.material_indices.size());
	m_hiprt_scene.material_indices.upload_data(scene.material_indices.data());

	// Uploading the materials after the textures have been parsed because texture
	// parsing can modify the materials (emission of constant textures are stored in the
	// material directly for example) so we need to wait for the end of texture parsing
	// to upload the materials
	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_MATERIALS, ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_MATERIALS, [this, &scene]() 
	{
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		std::vector<DevicePackedTexturedMaterial> packed_gpu_materials(scene.materials.size());
		for (int i = 0; i < scene.materials.size(); i++)
			packed_gpu_materials[i] = scene.materials[i].pack_to_GPU();

		m_hiprt_scene.materials_buffer.resize(scene.materials.size());
		m_hiprt_scene.materials_buffer.upload_data(packed_gpu_materials.data());

		// Computing the opaqueness of materials i.e. whether or not they are FULLY opaque
		std::vector<unsigned char> material_opaque(scene.materials.size());
		for (int i = 0; i < scene.materials.size(); i++)
			material_opaque[i] = scene.material_has_opaque_base_color_texture[i] && scene.materials[i].alpha_opacity == 1.0f;
		m_hiprt_scene.material_opaque.resize(material_opaque.size());
		m_hiprt_scene.material_opaque.upload_data(material_opaque);
		m_hiprt_scene.material_has_opaque_base_color_texture = scene.material_has_opaque_base_color_texture;

		m_hiprt_scene.texcoords_buffer.resize(scene.texcoords.size());
		m_hiprt_scene.texcoords_buffer.upload_data(scene.texcoords.data());
	});

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_TEXTURES, ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_TEXTURES, [this, &scene]() {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		if (scene.textures.size() > 0)
		{
			std::vector<oroTextureObject_t> oro_textures(scene.textures.size());
			m_hiprt_scene.orochi_materials_textures.reserve(scene.textures.size());
			for (int i = 0; i < scene.textures.size(); i++)
			{
				if (scene.textures[i].width == 0 || scene.textures[i].height == 0)
				{
					// It can happen that for emissive textures for example, we had a texture but its color is constant.
					// As a result, we have not read the texture but rather just stored the constant emissive color in the
					// emission filed of the material so we have no texture to read here

					// The shader will never read from that texture (because the texture index of the material has been set to -1)
					// so we set it to nullptr
					oro_textures[i] = nullptr;

					continue;
				}

				// We need to keep the texture alive so they are not destroyed when returning from 
				// this function so we're adding them to a member buffer
				m_hiprt_scene.orochi_materials_textures.push_back(OrochiTexture(scene.textures[i]));

				oro_textures[i] = m_hiprt_scene.orochi_materials_textures.back().get_device_texture();
			}

			m_hiprt_scene.gpu_materials_textures.resize(oro_textures.size());
			m_hiprt_scene.gpu_materials_textures.upload_data(oro_textures.data());
		}
	});

	ThreadManager::add_dependency(ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_UPLOAD_EMISSIVE_TRIANGLES, [this, &scene]() {
		m_hiprt_scene.emissive_triangles_count = scene.emissive_triangle_indices.size();
		if (m_hiprt_scene.emissive_triangles_count > 0)
		{
			OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

			m_hiprt_scene.emissive_triangles_indices.resize(scene.emissive_triangle_indices.size());
			m_hiprt_scene.emissive_triangles_indices.upload_data(scene.emissive_triangle_indices.data());
		}
	});
}

void GPURenderer::rebuild_renderer_bvh(hiprtBuildFlags build_flags, bool do_compaction)
{
	m_hiprt_scene.geometry.build_bvh(build_flags, do_compaction, m_main_stream);
}

void GPURenderer::set_scene(const Scene& scene)
{
	set_hiprt_scene_from_scene(scene);
	setup_nee_plus_plus_from_scene(scene);

	m_original_materials = scene.materials;
	m_current_materials = scene.materials;
	m_parsed_scene_metadata = scene.metadata;
}

void GPURenderer::set_envmap(const Image32Bit& envmap_image, const std::string& envmap_filepath)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_SET_ENVMAP, ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD);
	ThreadManager::start_thread(ThreadManager::RENDERER_SET_ENVMAP, [this, &envmap_image, &envmap_filepath]() {
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_hiprt_orochi_ctx->orochi_ctx));

		if (envmap_image.width == 0 || envmap_image.height == 0)
		{
			if (m_render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
				// We were going for the envmap but it's not available so defaulting to
				// uniform lighting instead
				m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;

			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Empty envmap set on the GPURenderer... Defaulting to uniform ambient light instead.");

			return;
		}

		m_envmap.init_from_image(envmap_image, envmap_filepath);
		m_envmap.recompute_sampling_data_structure(this, &envmap_image);

		m_render_data.world_settings.envmap = m_envmap.get_packed_data_pointer();
		m_render_data.world_settings.envmap_width = m_envmap.get_width();
		m_render_data.world_settings.envmap_height = m_envmap.get_height();
		// We found an envmap so let's use it
		m_render_data.world_settings.ambient_light_type = AmbientLightType::ENVMAP;

#if EnvmapSamplingStrategy == ESS_BINARY_SEARCH
		m_render_data.world_settings.envmap_cdf = m_envmap.get_cdf_device_pointer();

		m_render_data.world_settings.alias_table_probas = nullptr;
		m_render_data.world_settings.alias_table_alias = nullptr;
#elif EnvmapSamplingStrategy == ESS_ALIAS_TABLE
		m_render_data.world_settings.envmap_cdf = nullptr;

		m_envmap.get_alias_table_device_pointers(m_render_data.world_settings.alias_table_probas, m_render_data.world_settings.alias_table_alias);
#endif
	});
}

bool GPURenderer::has_envmap()
{
	return m_render_data.world_settings.envmap_height != 0 && m_render_data.world_settings.envmap_width != 0;
}

const std::vector<CPUMaterial>& GPURenderer::get_original_materials()
{
	return m_original_materials;
}

const std::vector<CPUMaterial>& GPURenderer::get_current_materials()
{
	return m_current_materials;
}

const std::vector<std::string>& GPURenderer::get_material_names()
{
	return m_parsed_scene_metadata.material_names;
}

void GPURenderer::update_all_materials(std::vector<CPUMaterial>& materials)
{
	m_current_materials = materials;

	std::vector<unsigned char> new_opacity(materials.size());
	std::vector<DevicePackedTexturedMaterial> packed_gpu_materials(materials.size());
	for (int i = 0; i < materials.size(); i++)
	{
		packed_gpu_materials[i] = materials[i].pack_to_GPU();

		// The material is fully opaque if its base color texture is fully opaque
		// and if the alpha opacity is fully opaque too (1.0f)
		new_opacity[i] = materials[i].alpha_opacity == 1.0f && m_hiprt_scene.material_has_opaque_base_color_texture[i];
	}

	// Because the materials have changed, reuploading the "precomputed oapcity" of the materials
	m_hiprt_scene.material_opaque.upload_data(new_opacity);
	m_hiprt_scene.materials_buffer.upload_data(packed_gpu_materials);
}

void GPURenderer::update_one_material(CPUMaterial& material, int material_index)
{
	m_current_materials[material_index] = material;

	DevicePackedTexturedMaterial packed_gpu_material = material.pack_to_GPU();
	// The material is fully opaque if its base color texture is fully opaque
	// and if the alpha opacity is fully opaque too (1.0f)
	unsigned char new_opacity = material.alpha_opacity == 1.0f && m_hiprt_scene.material_has_opaque_base_color_texture[material_index];

	// Because the materials have changed, reuploading the "precomputed oapcity" of the materials
	m_hiprt_scene.material_opaque.upload_data_partial(material_index, &new_opacity, 1);
	m_hiprt_scene.materials_buffer.upload_data_partial(material_index, &packed_gpu_material, 1);
}

const std::vector<BoundingBox>& GPURenderer::get_mesh_bounding_boxes()
{
	return m_parsed_scene_metadata.mesh_bounding_boxes;
}

const std::vector<std::string>& GPURenderer::get_mesh_names()
{
	return m_parsed_scene_metadata.mesh_names;
}

const std::vector<int>& GPURenderer::get_mesh_material_indices()
{
	return m_parsed_scene_metadata.mesh_material_indices;
}

size_t GPURenderer::get_ray_volume_state_byte_size()
{
	OrochiBuffer<size_t> out_size_buffer(1);
	size_t* out_size_buffer_pointer = out_size_buffer.get_device_pointer();

	ThreadManager::join_threads(ThreadManager::COMPILE_RAY_VOLUME_STATE_SIZE_KERNEL_KEY);

	void* launch_args[] = { &out_size_buffer_pointer };
	m_ray_volume_state_byte_size_kernel.launch_synchronous(1, 1, 1, 1, launch_args, 0);
	OROCHI_CHECK_ERROR(oroStreamSynchronize(0));

	return out_size_buffer.download_data()[0];
}

void GPURenderer::resize_g_buffer_ray_volume_states()
{
	synchronize_kernel();

	m_g_buffer.ray_volume_states.resize(m_render_resolution.x * m_render_resolution.y, get_ray_volume_state_byte_size());
	if (m_render_data.render_settings.use_prev_frame_g_buffer())
		m_g_buffer_prev_frame.ray_volume_states.resize(m_render_resolution.x * m_render_resolution.y, get_ray_volume_state_byte_size());

	m_render_data_buffers_invalidated = true;
}

Camera& GPURenderer::get_camera()
{
	return m_camera;
}

CameraAnimation& GPURenderer::get_camera_animation()
{
	return m_camera_animation;
}

RendererEnvmap& GPURenderer::get_envmap()
{
	return m_envmap;
}

void GPURenderer::set_camera(const Camera& camera)
{
	m_camera = camera;
	m_camera_animation.set_camera(&m_camera);
}

void GPURenderer::translate_camera_view(glm::vec3 translation)
{
	m_camera.translate(translation);
}

void GPURenderer::rotate_camera_view(glm::vec3 rotation_angles)
{
	m_camera.rotate(rotation_angles);
}

void GPURenderer::zoom_camera_view(float offset)
{
	m_camera.zoom(offset);
}

RendererAnimationState& GPURenderer::get_animation_state()
{
	return m_animation_state;
}
