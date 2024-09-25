	/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "Renderer/GPURenderer.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"

#include <Orochi/OrochiUtils.h>

const std::string GPURenderer::CAMERA_RAYS_KERNEL_ID = "Camera Rays";
const std::string GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID = "ReSTIR DI Initial Candidates";
const std::string GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID = "ReSTIR DI Temporal Reuse";
const std::string GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID = "ReSTIR DI Spatial Reuse";
const std::string GPURenderer::PATH_TRACING_KERNEL_ID = "Path Tracing";

const std::string GPURenderer::CAMERA_RAYS_KERNEL_FUNC_NAME = "CameraRays";
const std::string GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_FUNC_NAME = "ReSTIR_DI_InitialCandidates";
const std::string GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_FUNC_NAME = "ReSTIR_DI_TemporalReuse";
const std::string GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_FUNC_NAME = "ReSTIR_DI_SpatialReuse";
const std::string GPURenderer::PATH_TRACING_KERNEL_FUNC_NAME = "FullPathTracer";
const std::string GPURenderer::RAY_VOLUME_STATE_SIZE_KERNEL_FUNC_NAME = "RayVolumeStateSize";

const std::string GPURenderer::FULL_FRAME_TIME_KEY = "FullFrameTime";

const std::string GPURenderer::KERNEL_FILES[] = 
{
	DEVICE_KERNELS_DIRECTORY "/CameraRays.h", 
	DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_InitialCandidates.h",
	DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_TemporalReuse.h",
	DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_SpatialReuse.h",
	DEVICE_KERNELS_DIRECTORY "/FullPathTracer.h", 
	DEVICE_KERNELS_DIRECTORY "/Utils/RayVolumeStateSize.h", 
};

const std::string GPURenderer::KERNEL_FUNCTIONS[] = 
{ 
	CAMERA_RAYS_KERNEL_FUNC_NAME,
	RESTIR_DI_INITIAL_CANDIDATES_KERNEL_FUNC_NAME,
	RESTIR_DI_TEMPORAL_REUSE_KERNEL_FUNC_NAME,
	RESTIR_DI_SPATIAL_REUSE_KERNEL_FUNC_NAME,
	PATH_TRACING_KERNEL_FUNC_NAME,
	RAY_VOLUME_STATE_SIZE_KERNEL_FUNC_NAME,
};

const std::vector<std::string> GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS = 
{ 
	KERNEL_COMPILER_ADDITIONAL_INCLUDE, 
	DEVICE_INCLUDES_DIRECTORY, 
	OROCHI_INCLUDES_DIRECTORY, 
	"./" 
};

GPURenderer::GPURenderer(std::shared_ptr<HIPRTOrochiCtx> hiprt_oro_ctx)
{
	m_rng.m_state.seed = 42;

	// Creating buffers
	m_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_denoised_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_normals_AOV_buffer = std::make_shared<OpenGLInteropBuffer<float3>>();
	m_albedo_AOV_buffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	m_pixels_converged_sample_count_buffer = std::make_shared<OpenGLInteropBuffer<int>>();
	
	m_hiprt_orochi_ctx = hiprt_oro_ctx;	
	m_device_properties = m_hiprt_orochi_ctx->device_properties;

	setup_kernels();

	m_ms_time_per_pass["All"] = 0.0f;
	for (std::string pass : KERNEL_FUNCTIONS)
		m_ms_time_per_pass[pass] = 0.0f;

	OROCHI_CHECK_ERROR(oroStreamCreate(&m_main_stream));

	// Buffer that keeps track of whether at least one ray is still alive or not
	unsigned char true_data = 1;
	m_still_one_ray_active_buffer.resize(1);
	m_still_one_ray_active_buffer.upload_data(&true_data);
	m_pixels_converged_count_buffer.resize(1);

	OROCHI_CHECK_ERROR(oroEventCreate(&m_restir_di_state.spatial_reuse_time_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_restir_di_state.spatial_reuse_time_stop));

	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_start_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_stop_event));
}

void GPURenderer::setup_kernels()
{
	/*GPUKernel test_kernel;
	test_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/RegistersTest.h");
	test_kernel.set_kernel_function_name("TestFunction");
	test_kernel.get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	test_kernel.compile(m_hiprt_orochi_ctx);*/

	m_global_compiler_options = std::make_shared<GPUKernelCompilerOptions>();
	// Adding hardware acceleration by default if supported
	m_global_compiler_options->set_macro_value("__USE_HWI__", device_supports_hardware_acceleration() == HardwareAccelerationSupport::SUPPORTED);

	// List of options that will be specific to each kernel. We don't want these options
	// to be synchronized between kernels
	std::unordered_set<std::string> options_excluded_from_synchro =
	{
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_GLOBAL_RAYS,
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SHADOW_RAYS,
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS,
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS
	};

	// Some default values are set for SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS and SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS
	// which I found work approximately well in terms of performance on various scenes (not perfect though and, on top of not 
	// being perfect, this was measured on a 7900XTX with hardware accelerated ray tracing so... your mileage in terms of what 
	// numbers are the best may vary. A lot.)
	
	// Configuring the kernels
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES[0]);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[0]);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, options_excluded_from_synchro);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	// The camera rays kernel doesn't need shadow rays so let's not use shared memory for that
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SHADOW_RAYS, KERNEL_OPTION_FALSE);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS, 0);
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS, 48);

	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES[1]);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[1]);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, options_excluded_from_synchro);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_GLOBAL_RAYS, KERNEL_OPTION_FALSE);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SHADOW_RAYS, KERNEL_OPTION_TRUE);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS, 0);
	m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS, 1);

	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES[2]);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[2]);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, options_excluded_from_synchro);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_GLOBAL_RAYS, KERNEL_OPTION_FALSE);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SHADOW_RAYS, KERNEL_OPTION_TRUE);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS, 0);
	m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS, 16);

	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES[3]);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[3]);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, options_excluded_from_synchro);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_GLOBAL_RAYS, KERNEL_OPTION_FALSE);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SHADOW_RAYS, KERNEL_OPTION_TRUE);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS, 0);
	m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS, 0);

	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].set_kernel_file_path(GPURenderer::KERNEL_FILES[4]);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[4]);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].synchronize_options_with(*m_global_compiler_options, options_excluded_from_synchro);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_GLOBAL_RAYS, 12);
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE_SHADOW_RAYS, 0);

	// Configuring the kernel that will be used to retrieve the size of the RayVolumeState structure.
	// This size will be needed to resize the 'ray_volume_states' buffer in the GBuffer if the nested dielectrics
	// stack size changes
	m_ray_volume_state_byte_size_kernel.set_kernel_file_path(GPURenderer::KERNEL_FILES[5]);
	m_ray_volume_state_byte_size_kernel.set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[5]);
	m_ray_volume_state_byte_size_kernel.get_kernel_options().set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);
	// Synchronizing here so that when the nested dielectrics stack size of the
	// global kernel option changes, this kernel has the size change too and can
	// be recompiled accordingly
	m_ray_volume_state_byte_size_kernel.synchronize_options_with(*m_global_compiler_options, options_excluded_from_synchro);
	m_ray_volume_state_byte_size_kernel.compile_silent(m_hiprt_orochi_ctx);

	// Compiling kernels
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID]), m_hiprt_orochi_ctx);
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]), m_hiprt_orochi_ctx);
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]), m_hiprt_orochi_ctx);
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]), m_hiprt_orochi_ctx);
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID]), m_hiprt_orochi_ctx);
}

void GPURenderer::update()
{
	internal_update_clear_device_status_buffers();
	internal_update_prev_frame_g_buffer();
	internal_update_adaptive_sampling_buffers();
	internal_update_restir_di_buffers();
	internal_update_global_stack_buffer();

	update_render_data();

	// Resetting this flag as this is a new frame
	m_render_data.render_settings.do_update_status_buffers = false;

	if (!m_render_data.render_settings.accumulate)
		m_render_data.render_settings.sample_number = 0;
}

void GPURenderer::copy_status_buffers()
{
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.one_ray_active, m_still_one_ray_active_buffer.get_device_pointer(), sizeof(unsigned char), oroMemcpyDeviceToHost));
	OROCHI_CHECK_ERROR(oroMemcpy(&m_status_buffers_values.pixel_converged_count, m_pixels_converged_count_buffer.get_device_pointer(), sizeof(unsigned int), oroMemcpyDeviceToHost));
}

void GPURenderer::internal_update_clear_device_status_buffers()
{
	unsigned char false_data = false;
	unsigned int zero_data = 0;
	// Uploading false to reset the flag
	m_still_one_ray_active_buffer.upload_data(&false_data);
	// Resetting the counter of pixels converged to 0
	m_pixels_converged_count_buffer.upload_data(&zero_data);
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
		bool prev_frame_g_buffer_needs_resize = m_g_buffer_prev_frame.cameray_ray_hit.get_element_count() == 0;

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

		if (m_g_buffer_prev_frame.cameray_ray_hit.get_element_count() > 0)
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
		{
			// If one of the two buffers is going to be resized, synchronizing because we don't want
			// to resize the buffers if we're currently rendering a frame
			synchronize_kernel();

			// At least on buffer is going to be resized so buffers are invalidated
			m_render_data_buffers_invalidated = true;
		}

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
		{
			// If one of the buffers isn't freed already, we're going to free it. In this case, we need to synchronize to avoid
			// freeing a buffer that the renderer is actively using in the frame it is rendering right now
			synchronize_kernel();

			m_render_data_buffers_invalidated = true;
		}

		m_pixels_squared_luminance_buffer.free();
		m_pixels_sample_count_buffer.free();
		m_pixels_converged_sample_count_buffer->free();
	}
}

void GPURenderer::internal_update_restir_di_buffers()
{
	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		// ReSTIR DI enabled
		bool initial_candidates_reservoir_needs_resize = m_restir_di_state.initial_candidates_reservoirs.get_element_count() == 0;
		bool spatial_output_1_needs_resize = m_restir_di_state.spatial_reuse_output_1.get_element_count() == 0;
		bool spatial_output_2_needs_resize = m_restir_di_state.spatial_reuse_output_2.get_element_count() == 0;

		if (initial_candidates_reservoir_needs_resize || spatial_output_1_needs_resize || spatial_output_2_needs_resize)
		{
			// Synchronizing because we don't want to resize the buffer while the renderer is rendering a frame
			synchronize_kernel();

			// At least on buffer is going to be resized so buffers are invalidated
			m_render_data_buffers_invalidated = true;
		}

		if (initial_candidates_reservoir_needs_resize)
			m_restir_di_state.initial_candidates_reservoirs.resize(m_render_resolution.x * m_render_resolution.y);

		if (spatial_output_1_needs_resize)
			m_restir_di_state.spatial_reuse_output_1.resize(m_render_resolution.x * m_render_resolution.y);

		if (spatial_output_2_needs_resize)
			m_restir_di_state.spatial_reuse_output_2.resize(m_render_resolution.x * m_render_resolution.y);
	}
	else
	{
		// ReSTIR DI disabled, we're going to free the buffers if that's not already done
		if (m_restir_di_state.initial_candidates_reservoirs.get_element_count() > 0
			|| m_restir_di_state.spatial_reuse_output_1.get_element_count() > 0 ||
			m_restir_di_state.spatial_reuse_output_2.get_element_count() > 0)
		{
			// If one of the buffers isn't freed already, we're going to free it. In this case, we need to synchronize to avoid
			// freeing a buffer that the renderer is actively using in the frame it is rendering right now
			synchronize_kernel();

			m_render_data_buffers_invalidated = true;
		}

		m_restir_di_state.initial_candidates_reservoirs.free();
		m_restir_di_state.spatial_reuse_output_1.free();
		m_restir_di_state.spatial_reuse_output_2.free();
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
		{
			// Creating the global stack buffer for BVH traversal if it doesn't exist already
			hiprtGlobalStackBufferInput stackBufferInput
			{
				hiprtStackTypeGlobal,
				hiprtStackEntryTypeInteger,
				static_cast<uint32_t>(m_render_data.global_traversal_stack_buffer_size),
				static_cast<uint32_t>(std::ceil(m_render_resolution.x / 8.0f) * 8 * 8 * std::ceil(m_render_resolution.y / 8.0f))
			};

			if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
				// Freeing if the buffer is already created
				HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));

			HIPRT_CHECK_ERROR(hiprtCreateGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, stackBufferInput, m_render_data.global_traversal_stack_buffer));
		}
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
		global_stack_buffer_needed |= name_to_kernel.second.get_kernel_options().get_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_GLOBAL_RAYS) == KERNEL_OPTION_TRUE;
		global_stack_buffer_needed |= name_to_kernel.second.get_kernel_options().get_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SHADOW_RAYS) == KERNEL_OPTION_TRUE;

		if (global_stack_buffer_needed)
			return true;
	}

	return false;
}

void GPURenderer::render()
{
	// Making sure kernels are compiled
	ThreadManager::join_threads(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY);

	int tile_size_x = 8;
	int tile_size_y = 8;

	int2 nb_groups;
	nb_groups.x = std::ceil(m_render_resolution.x / (float)tile_size_x);
	nb_groups.y = std::ceil(m_render_resolution.y / (float)tile_size_y);

	map_buffers_for_render();
	
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
		// If we had requested a temporal buffers clear, this has be done by this frame so we can
		// now reset the flag
		m_render_data.render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = false;

		// Updating the cameras
		m_previous_frame_camera = m_camera;
	}

	// Recording GPU frame time stop timestamp and computing the frame time
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, m_main_stream));

	GPUKernel::ComputeElapsedTimeCallbackData* elapsed_time_data = new GPUKernel::ComputeElapsedTimeCallbackData;
	elapsed_time_data->start = m_frame_start_event;
	elapsed_time_data->end = m_frame_stop_event;
	elapsed_time_data->elapsed_time_out = &m_ms_time_per_pass["All"];

	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, GPUKernel::compute_elapsed_time_callback, elapsed_time_data));

	// Updating the times per passes
	m_ms_time_per_pass[GPURenderer::CAMERA_RAYS_KERNEL_ID] = m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].get_last_execution_time();
	m_ms_time_per_pass[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID] = m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].get_last_execution_time();
	m_ms_time_per_pass[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID] = m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].get_last_execution_time();
	// RESTIR_DI_SPATIAL_REUSE_KERNEL_ID
	// - The spatial reuse time is handled directly when the spatial reuse kernel is launched because
	//	there may be multiple spatial reuse passes in which case, we need to sum the times of all the passes
	//	but get_last_execution_time() doesn't support that, it only contains the time of the *last* pass
	m_ms_time_per_pass[GPURenderer::PATH_TRACING_KERNEL_ID] = m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].get_last_execution_time();

	m_was_last_frame_low_resolution = m_render_data.render_settings.do_render_low_resolution();
}

void GPURenderer::launch_camera_rays()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	m_render_data.random_seed = m_rng.xorshift32();
	m_kernels[GPURenderer::CAMERA_RAYS_KERNEL_ID].launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::launch_ReSTIR_DI()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		// If ReSTIR DI is enabled

		configure_ReSTIR_DI_initial_pass();
		m_kernels[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID].launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);

		if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
		{
			configure_ReSTIR_DI_temporal_pass();

			m_kernels[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID].launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
		}


		if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
		{
			// Emitting an event for timing all the spatial reuse passes combined
			OROCHI_CHECK_ERROR(oroEventRecord(m_restir_di_state.spatial_reuse_time_start, m_main_stream));

			for (int spatial_reuse_pass = 0; spatial_reuse_pass < m_render_data.render_settings.restir_di_settings.spatial_pass.number_of_passes; spatial_reuse_pass++)
			{
				configure_ReSTIR_DI_spatial_pass(spatial_reuse_pass);

				m_kernels[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID].launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
			}

			// Emitting the stop event
			OROCHI_CHECK_ERROR(oroEventRecord(m_restir_di_state.spatial_reuse_time_stop, m_main_stream));

			GPUKernel::ComputeElapsedTimeCallbackData* elapsed_time_data = new GPUKernel::ComputeElapsedTimeCallbackData;
			elapsed_time_data->start = m_restir_di_state.spatial_reuse_time_start;
			elapsed_time_data->end = m_restir_di_state.spatial_reuse_time_stop;
			elapsed_time_data->elapsed_time_out = &m_ms_time_per_pass[GPURenderer::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID];

			// Computing the time elapsed for all spatial reuse passes
			OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, GPUKernel::compute_elapsed_time_callback, elapsed_time_data));
		}

		configure_ReSTIR_DI_output_buffer();

		m_restir_di_state.odd_frame = !m_restir_di_state.odd_frame;
	}
}

void GPURenderer::configure_ReSTIR_DI_initial_pass()
{
	m_render_data.random_seed = m_rng.xorshift32();
	m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.get_device_pointer();
}

void GPURenderer::configure_ReSTIR_DI_temporal_pass()
{
	m_render_data.random_seed = m_rng.xorshift32();
	m_render_data.render_settings.restir_di_settings.temporal_pass.permutation_sampling_random_bits = m_rng.xorshift32();

	// The input of the temporal pass is the output of last frame ReSTIR (and also the initial candidates but this is implicit
	// and "hardcoded in the shader"
	m_render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.restir_output_reservoirs;

	if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
		m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_restir_di_state.initial_candidates_reservoirs.get_device_pointer();
	// If we're going to do spatial reuse, reuse the initial candidate reservoirs to store the output of the temporal pass.
	// The spatial reuse pass will read form that buffer
	else
	{
		// Else, no spatial reuse, the output of the temporal pass is going to be in its own buffer.
		// Alternatively using spatial_reuse_output_1 and spatial_reuse_output_2 to avoid race conditions
		if (m_restir_di_state.odd_frame)
			m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
		else
			m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_restir_di_state.spatial_reuse_output_2.get_device_pointer();
	}
}

void GPURenderer::configure_ReSTIR_DI_spatial_pass(int spatial_pass_index)
{
	m_render_data.random_seed = m_rng.xorshift32();
	m_render_data.render_settings.restir_di_settings.spatial_pass.spatial_pass_index = spatial_pass_index;

	if (spatial_pass_index == 0)
	{
		if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
			// For the first spatial reuse pass, we hardcode reading from the output of the temporal pass and storing into 'spatial_reuse_output_1'
			m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs;
		else
			// If there is no temporal reuse pass, using the initial candidates as the input to the spatial reuse pass
			m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;

		m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
	}
	else
	{
		// And then, starting at the second spatial reuse pass, we read from the output of the previous spatial pass and store
		// in either spatial_reuse_output_1 or spatial_reuse_output_2, depending on which one isn't the input (we don't
		// want to store in the same buffers that is used for output because that's a race condition so
		// we're ping-ponging between the two outputs of the spatial reuse pass)

		if ((spatial_pass_index & 1) == 0)
		{
			m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_restir_di_state.spatial_reuse_output_2.get_device_pointer();
			m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
		}
		else
		{
			m_render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
			m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_restir_di_state.spatial_reuse_output_2.get_device_pointer();

		}
	}
}

void GPURenderer::configure_ReSTIR_DI_output_buffer()
{
	// Keeping in mind which was the buffer used last for the output of the spatial reuse pass as this is the buffer that
		// we're going to use as the input to the temporal reuse pass of the next frame
	if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
		// If there was spatial reuse, using the output of the spatial reuse pass as the input of the temporal
		// pass of next frame
		m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs;
	else if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
		// If there was a temporal reuse pass, using that output as the input of the next temporal reuse pass
		m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs;
	else
		// No spatial or temporal, the output of ReSTIR is just the output of the initial candidates pass
		m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;
}

void GPURenderer::launch_path_tracing()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	m_render_data.random_seed = m_rng.xorshift32();
	m_kernels[GPURenderer::PATH_TRACING_KERNEL_ID].launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::synchronize_kernel()
{
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_main_stream));
}

bool GPURenderer::frame_render_done()
{
	return oroStreamQuery(m_main_stream) == oroSuccess;
}

bool GPURenderer::was_last_frame_low_resolution()
{
	return m_was_last_frame_low_resolution;
}

void GPURenderer::resize(int new_width, int new_height)
{
	m_render_resolution = make_int2(new_width, new_height);

	synchronize_kernel();
	unmap_buffers();

	m_framebuffer->resize(new_width * new_height);
	m_denoised_framebuffer->resize(new_width * new_height);
	m_normals_AOV_buffer->resize(new_width * new_height);
	m_albedo_AOV_buffer->resize(new_width * new_height);

	m_g_buffer.resize(new_width * new_height, get_ray_volume_state_byte_size());

	if (m_render_data.render_settings.use_prev_frame_g_buffer(this))
		m_g_buffer_prev_frame.resize(new_width * new_height, get_ray_volume_state_byte_size());

	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		m_pixels_squared_luminance_buffer.resize(new_width * new_height);
		m_pixels_sample_count_buffer.resize(new_width * new_height);
		m_pixels_converged_sample_count_buffer->resize(new_width * new_height);
	}

	if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		m_restir_di_state.initial_candidates_reservoirs.resize(new_width * new_height);
		m_restir_di_state.spatial_reuse_output_2.resize(new_width * new_height);
		m_restir_di_state.spatial_reuse_output_1.resize(new_width * new_height);
	}

	m_pixel_active.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.set_aspect(new_aspect);

	if (needs_global_bvh_stack_buffer())
	{
		// Resizing the global stack buffer for BVH traversal
		hiprtGlobalStackBufferInput stackBufferInput
		{
			hiprtStackTypeGlobal,
			hiprtStackEntryTypeInteger,
			static_cast<uint32_t>(m_render_data.global_traversal_stack_buffer_size),
			static_cast<uint32_t>(std::ceil(m_render_resolution.x / 8.0f) * 8 * 8 * std::ceil(m_render_resolution.y / 8.0f))
		};

		if (m_render_data.global_traversal_stack_buffer.stackData != nullptr)
			// Freeing if the buffer already exists
			HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, m_render_data.global_traversal_stack_buffer));

		HIPRT_CHECK_ERROR(hiprtCreateGlobalStackBuffer(m_hiprt_orochi_ctx->hiprt_ctx, stackBufferInput, m_render_data.global_traversal_stack_buffer));
	}

	m_render_data_buffers_invalidated = true;
}

void GPURenderer::map_buffers_for_render()
{
	m_render_data.buffers.pixels = m_framebuffer->map_no_error();
	m_render_data.aux_buffers.denoiser_normals = m_normals_AOV_buffer->map_no_error();
	m_render_data.aux_buffers.denoiser_albedo = m_albedo_AOV_buffer->map_no_error();
	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		m_render_data.aux_buffers.pixel_converged_sample_count = m_pixels_converged_sample_count_buffer->map_no_error();
}

void GPURenderer::unmap_buffers()
{
	m_framebuffer->unmap();
	m_normals_AOV_buffer->unmap();
	m_albedo_AOV_buffer->unmap();
	m_pixels_converged_sample_count_buffer->unmap();
}


std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_color_framebuffer()
{
	return m_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoised_framebuffer()
{
	return m_denoised_framebuffer;
}

std::shared_ptr<OpenGLInteropBuffer<float3>> GPURenderer::get_denoiser_normals_AOV_buffer()
{
	return m_normals_AOV_buffer;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GPURenderer::get_denoiser_albedo_AOV_buffer()
{
	return m_albedo_AOV_buffer;
}

std::shared_ptr<OpenGLInteropBuffer<int>>& GPURenderer::get_pixels_converged_sample_count_buffer()
{
	return m_pixels_converged_sample_count_buffer;
}

const StatusBuffersValues& GPURenderer::get_status_buffer_values() const
{
	return m_status_buffers_values;
}

HIPRTRenderSettings& GPURenderer::get_render_settings()
{
	return m_render_data.render_settings;
}

WorldSettings& GPURenderer::get_world_settings()
{
	return m_render_data.world_settings;
}

HIPRTRenderData& GPURenderer::get_render_data()
{
	return m_render_data;
}

oroDeviceProp GPURenderer::get_device_properties()
{
	return m_device_properties;
}

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

void GPURenderer::recompile_kernels(bool use_cache)
{
	synchronize_kernel();

	for (auto& name_to_kenel : m_kernels)
		name_to_kenel.second.compile(m_hiprt_orochi_ctx, use_cache);
	m_ray_volume_state_byte_size_kernel.compile_silent(m_hiprt_orochi_ctx);
}

std::map<std::string, GPUKernel>& GPURenderer::get_kernels()
{
	return m_kernels;
}

float GPURenderer::get_render_pass_time(const std::string& key)
{
	return m_ms_time_per_pass[key];
}

void GPURenderer::reset_frame_times()
{
	m_ms_time_per_pass["All"] = 0.0f;
	for (std::string pass : KERNEL_FUNCTIONS)
		m_ms_time_per_pass[pass] = 0.0f;
}

void GPURenderer::reset(std::shared_ptr<ApplicationSettings> application_settings)
{
	if (m_render_data.render_settings.accumulate)
	{
		// Only resetting the seed for deterministic rendering if we're accumulating.
		// If we're not accumulating, we want each frame of the render to be different
		// so we don't get into that if block and we don't reset the seed
		m_rng.m_state.seed = 42;

		m_restir_di_state.odd_frame = false;
	
		if (application_settings->auto_sample_per_frame)
			m_render_data.render_settings.samples_per_frame = 1;
	}

	m_render_data.render_settings.denoiser_AOV_accumulation_counter = 0;
	m_render_data.render_settings.sample_number = 0;
	m_render_data.render_settings.need_to_reset = true;

	reset_frame_times();
	internal_clear_m_status_buffers();
}

void GPURenderer::update_render_data()
{
	// Always updating the random seed
	m_render_data.random_seed = m_rng.xorshift32();

	if (m_render_data_buffers_invalidated)
	{
		m_render_data.geom = m_hiprt_scene.geometry.m_geometry;

		m_render_data.buffers.triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.geometry.m_mesh.triangleIndices);
		m_render_data.buffers.vertices_positions = reinterpret_cast<float3*>(m_hiprt_scene.geometry.m_mesh.vertices);
		m_render_data.buffers.has_vertex_normals = reinterpret_cast<unsigned char*>(m_hiprt_scene.has_vertex_normals.get_device_pointer());
		m_render_data.buffers.vertex_normals = reinterpret_cast<float3*>(m_hiprt_scene.vertex_normals.get_device_pointer());
		m_render_data.buffers.material_indices = reinterpret_cast<int*>(m_hiprt_scene.material_indices.get_device_pointer());
		m_render_data.buffers.materials_buffer = reinterpret_cast<RendererMaterial*>(m_hiprt_scene.materials_buffer.get_device_pointer());
		m_render_data.buffers.emissive_triangles_count = m_hiprt_scene.emissive_triangles_count;
		m_render_data.buffers.emissive_triangles_indices = reinterpret_cast<int*>(m_hiprt_scene.emissive_triangles_indices.get_device_pointer());

		m_render_data.buffers.material_textures = reinterpret_cast<oroTextureObject_t*>(m_hiprt_scene.materials_textures.get_device_pointer());
		m_render_data.buffers.texcoords = reinterpret_cast<float2*>(m_hiprt_scene.texcoords_buffer.get_device_pointer());
		m_render_data.buffers.textures_dims = reinterpret_cast<int2*>(m_hiprt_scene.textures_dims.get_device_pointer());

		m_render_data.g_buffer.materials = m_g_buffer.materials.get_device_pointer();
		m_render_data.g_buffer.geometric_normals = m_g_buffer.geometric_normals.get_device_pointer();
		m_render_data.g_buffer.shading_normals = m_g_buffer.shading_normals.get_device_pointer();
		m_render_data.g_buffer.view_directions = m_g_buffer.view_directions.get_device_pointer();
		m_render_data.g_buffer.first_hits = m_g_buffer.first_hits.get_device_pointer();
		m_render_data.g_buffer.camera_ray_hit = m_g_buffer.cameray_ray_hit.get_device_pointer();
		m_render_data.g_buffer.ray_volume_states = m_g_buffer.ray_volume_states.get_device_pointer();

		if (m_render_data.render_settings.use_prev_frame_g_buffer(this))
		{
			// Only setting the pointers of the buffers if we're actually using the g-buffer of the previous frame

			m_render_data.g_buffer_prev_frame.materials = m_g_buffer_prev_frame.materials.get_device_pointer();
			m_render_data.g_buffer_prev_frame.geometric_normals = m_g_buffer_prev_frame.geometric_normals.get_device_pointer();
			m_render_data.g_buffer_prev_frame.shading_normals = m_g_buffer_prev_frame.shading_normals.get_device_pointer();
			m_render_data.g_buffer_prev_frame.view_directions = m_g_buffer_prev_frame.view_directions.get_device_pointer();
			m_render_data.g_buffer_prev_frame.first_hits = m_g_buffer_prev_frame.first_hits.get_device_pointer();
			m_render_data.g_buffer_prev_frame.camera_ray_hit = m_g_buffer_prev_frame.cameray_ray_hit.get_device_pointer();
			m_render_data.g_buffer_prev_frame.ray_volume_states = m_g_buffer_prev_frame.ray_volume_states.get_device_pointer();
		}
		else
		{
			m_render_data.g_buffer_prev_frame.materials = nullptr;
			m_render_data.g_buffer_prev_frame.geometric_normals = nullptr;
			m_render_data.g_buffer_prev_frame.shading_normals = nullptr;
			m_render_data.g_buffer_prev_frame.view_directions = nullptr;
			m_render_data.g_buffer_prev_frame.first_hits = nullptr;
			m_render_data.g_buffer_prev_frame.camera_ray_hit = nullptr;
			m_render_data.g_buffer_prev_frame.ray_volume_states = nullptr;
		}

		if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
		{
			m_render_data.aux_buffers.pixel_sample_count = m_pixels_sample_count_buffer.get_device_pointer();
			m_render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance_buffer.get_device_pointer();
		}

		m_render_data.aux_buffers.pixel_active = m_pixel_active.get_device_pointer();
		m_render_data.aux_buffers.still_one_ray_active = m_still_one_ray_active_buffer.get_device_pointer();
		m_render_data.aux_buffers.stop_noise_threshold_converged_count = reinterpret_cast<AtomicType<unsigned int>*>(m_pixels_converged_count_buffer.get_device_pointer());

		// Setting the pointers for use in reset_render() in the camera rays kernel
		if (m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
		{
			m_render_data.aux_buffers.restir_reservoir_buffer_1 = m_restir_di_state.initial_candidates_reservoirs.get_device_pointer();
			m_render_data.aux_buffers.restir_reservoir_buffer_2 = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
			m_render_data.aux_buffers.restir_reservoir_buffer_3 = m_restir_di_state.spatial_reuse_output_2.get_device_pointer();

			// If we just got ReSTIR enabled back, setting this one arbitrarily and resetting its content
			std::vector<ReSTIRDIReservoir> empty_reservoirs(m_render_resolution.x * m_render_resolution.y, ReSTIRDIReservoir());
			m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
			m_restir_di_state.spatial_reuse_output_1.upload_data(empty_reservoirs);
		}
		else
		{
			// If ReSTIR DI is disabled, setting the pointers to nullptr so that the camera rays kernel
			// for example can detect that the buffers are freed and doesn't try to reset them or do
			// anything with them (which would lead to a crash since we would be accessing nullptr buffers)

			m_render_data.aux_buffers.restir_reservoir_buffer_1 = nullptr;
			m_render_data.aux_buffers.restir_reservoir_buffer_2 = nullptr;
			m_render_data.aux_buffers.restir_reservoir_buffer_3 = nullptr;
		}

		m_render_data_buffers_invalidated = false;
	}
}

void GPURenderer::set_hiprt_scene_from_scene(const Scene& scene)
{
	m_hiprt_scene.geometry.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	m_hiprt_scene.geometry.upload_indices(scene.triangle_indices);
	m_hiprt_scene.geometry.upload_vertices(scene.vertices_positions);
	m_hiprt_scene.geometry.build_bvh();

	m_hiprt_scene.has_vertex_normals.resize(scene.has_vertex_normals.size());
	m_hiprt_scene.has_vertex_normals.upload_data(scene.has_vertex_normals.data());

	m_hiprt_scene.vertex_normals.resize(scene.vertex_normals.size());
	m_hiprt_scene.vertex_normals.upload_data(scene.vertex_normals.data());

	m_hiprt_scene.material_indices.resize(scene.material_indices.size());
	m_hiprt_scene.material_indices.upload_data(scene.material_indices.data());

	m_hiprt_scene.materials_buffer.resize(scene.materials.size());
	m_hiprt_scene.materials_buffer.upload_data(scene.materials.data());

	m_hiprt_scene.emissive_triangles_count = scene.emissive_triangle_indices.size();
	if (m_hiprt_scene.emissive_triangles_count > 0)
	{
		m_hiprt_scene.emissive_triangles_indices.resize(scene.emissive_triangle_indices.size());
		m_hiprt_scene.emissive_triangles_indices.upload_data(scene.emissive_triangle_indices.data());
	}

	m_hiprt_scene.texcoords_buffer.resize(scene.texcoords.size());
	m_hiprt_scene.texcoords_buffer.upload_data(scene.texcoords.data());

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

		m_hiprt_scene.materials_textures.resize(oro_textures.size());
		m_hiprt_scene.materials_textures.upload_data(oro_textures.data());

		m_hiprt_scene.textures_dims.resize(scene.textures_dims.size());
		m_hiprt_scene.textures_dims.upload_data(scene.textures_dims.data());
	}
}

void GPURenderer::set_scene(const Scene& scene)
{
	set_hiprt_scene_from_scene(scene);

	m_materials = scene.materials;
	m_material_names = scene.material_names;
}

void GPURenderer::set_envmap(Image32Bit& envmap_image)
{
	ThreadManager::join_threads(ThreadManager::ENVMAP_LOAD_THREAD_KEY);

	if (envmap_image.width == 0 || envmap_image.height == 0)
	{
		m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;

		std::cerr << "Empty envmap set on the GPURenderer... Defaulting to uniform ambient light type" << std::endl;

		return;
	}

	m_envmap.init_from_image(envmap_image);
	m_envmap.compute_cdf(envmap_image);

	m_render_data.world_settings.envmap = m_envmap.get_device_texture();
	m_render_data.world_settings.envmap_width = m_envmap.width;
	m_render_data.world_settings.envmap_height = m_envmap.height;
	m_render_data.world_settings.envmap_cdf = m_envmap.get_cdf_device_pointer();
} 

bool GPURenderer::has_envmap()
{
	return m_render_data.world_settings.envmap_height != 0 && m_render_data.world_settings.envmap_width != 0;
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

size_t GPURenderer::get_ray_volume_state_byte_size()
{
	OrochiBuffer<size_t> out_size_buffer(1);
	size_t* out_size_buffer_pointer = out_size_buffer.get_device_pointer();

	void* launch_args[] = { &out_size_buffer_pointer };
	m_ray_volume_state_byte_size_kernel.launch(1, 1, 1, 1, launch_args, 0);
	oroStreamSynchronize(0);

	std::vector<size_t> ray_volume_state_size = out_size_buffer.download_data();
	return ray_volume_state_size[0];
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

void GPURenderer::set_camera(const Camera& camera)
{
	m_camera = camera;
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
