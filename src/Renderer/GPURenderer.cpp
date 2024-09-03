/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Renderer/GPURenderer.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"
#include "UI/ApplicationSettings.h"

#include <Orochi/OrochiUtils.h>

const std::string GPURenderer::PATH_TRACING_KERNEL = "FullPathTracer";
const std::string GPURenderer::CAMERA_RAYS_FUNC_NAME = "CameraRays";
const std::string GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME = "ReSTIR_DI_InitialCandidates";
const std::string GPURenderer::RESTIR_DI_TEMPORAL_REUSE_FUNC_NAME = "ReSTIR_DI_TemporalReuse";
const std::string GPURenderer::RESTIR_DI_SPATIAL_REUSE_FUNC_NAME = "ReSTIR_DI_SpatialReuse";
const std::string GPURenderer::FULL_FRAME_TIME_KEY = "FullFrameTime";

const std::string GPURenderer::KERNEL_FILES[] = 
{
	DEVICE_KERNELS_DIRECTORY "/FullPathTracer.h", 
	DEVICE_KERNELS_DIRECTORY "/CameraRays.h", 
	DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_InitialCandidates.h",
	DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_TemporalReuse.h",
	DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_SpatialReuse.h" 
};

const std::string GPURenderer::KERNEL_FUNCTIONS[] = 
{ 
	PATH_TRACING_KERNEL, 
	CAMERA_RAYS_FUNC_NAME, 
	RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME, 
	RESTIR_DI_TEMPORAL_REUSE_FUNC_NAME, 
	RESTIR_DI_SPATIAL_REUSE_FUNC_NAME 
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
	m_pixels_sample_count_buffer = std::make_shared<OpenGLInteropBuffer<int>>();
	
	m_hiprt_orochi_ctx = hiprt_oro_ctx;	
	m_device_properties = m_hiprt_orochi_ctx->device_properties;

	m_path_tracer_options = std::make_shared<GPUKernelCompilerOptions>();
	// Adding hardware acceleration by default if supported
	if (device_supports_hardware_acceleration() == HardwareAccelerationSupport::SUPPORTED)
		m_path_tracer_options->set_macro("__USE_HWI__", 1);
	else
		m_path_tracer_options->remove_macro("__USE_HWI__");
	m_path_tracer_options->set_additional_include_directories(GPURenderer::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS);

	// Configuring the kernels
	m_path_trace_pass.set_kernel_file_path(GPURenderer::KERNEL_FILES[0]);
	m_path_trace_pass.set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[0]);

	m_camera_ray_pass.set_kernel_file_path(GPURenderer::KERNEL_FILES[1]);
	m_camera_ray_pass.set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[1]);

	m_restir_initial_candidates_pass.set_kernel_file_path(GPURenderer::KERNEL_FILES[2]);
	m_restir_initial_candidates_pass.set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[2]);

	m_restir_temporal_reuse_pass.set_kernel_file_path(GPURenderer::KERNEL_FILES[3]);
	m_restir_temporal_reuse_pass.set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[3]);

	m_restir_spatial_reuse_pass.set_kernel_file_path(GPURenderer::KERNEL_FILES[4]);
	m_restir_spatial_reuse_pass.set_kernel_function_name(GPURenderer::KERNEL_FUNCTIONS[4]);

	// Compiling kernels
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_path_trace_pass), m_path_tracer_options, std::ref(m_hiprt_orochi_ctx->hiprt_ctx));
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_camera_ray_pass), m_path_tracer_options, std::ref(m_hiprt_orochi_ctx->hiprt_ctx));
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_restir_initial_candidates_pass), m_path_tracer_options, std::ref(m_hiprt_orochi_ctx->hiprt_ctx));
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_restir_temporal_reuse_pass), m_path_tracer_options, std::ref(m_hiprt_orochi_ctx->hiprt_ctx));
	ThreadManager::start_thread(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY, ThreadFunctions::compile_kernel, std::ref(m_restir_spatial_reuse_pass), m_path_tracer_options, std::ref(m_hiprt_orochi_ctx->hiprt_ctx));

	m_ms_time_per_pass["All"] = 0.0f;
	for (std::string pass : KERNEL_FUNCTIONS)
		m_ms_time_per_pass[pass] = 0.0f;

	OROCHI_CHECK_ERROR(oroStreamCreate(&m_main_stream));

	// Buffer that keeps track of whether at least one ray is still alive or not
	unsigned char true_data = 1;
	m_still_one_ray_active_buffer.resize(1);
	m_still_one_ray_active_buffer.upload_data(&true_data);
	m_pixels_converged_count_buffer.resize(1);

	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_start_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_frame_stop_event));
}

void GPURenderer::update()
{
	internal_update_clear_device_status_buffers();
	internal_update_adaptive_sampling_buffers();
	internal_update_restir_di_buffers();

	// Resetting this flag as this is a new frame
	m_render_data.render_settings.do_update_status_buffers = false;

	if (!m_render_data.render_settings.accumulate)
	{
		m_render_data.render_settings.sample_number = 0;
	}
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

void GPURenderer::internal_update_adaptive_sampling_buffers()
{
	bool buffers_needed = m_render_data.render_settings.has_access_to_adaptive_sampling_buffers();

	if (buffers_needed)
	{
		bool pixels_squared_luminance_needs_resize = m_pixels_squared_luminance_buffer.get_element_count() == 0;
		bool pixels_sample_count_needs_resize = m_pixels_sample_count_buffer->get_element_count() == 0;

		if (pixels_squared_luminance_needs_resize || pixels_sample_count_needs_resize)
			// If one of the two buffers is going to be resized, synchronizing because we don't want
			// to resize the buffers if we're currently rendering a frame
			synchronize_kernel();

		if (pixels_squared_luminance_needs_resize)
			// Only allocating if it isn't already
			m_pixels_squared_luminance_buffer.resize(m_render_resolution.x * m_render_resolution.y);

		if (pixels_sample_count_needs_resize)
			// Only allocating if it isn't already
			m_pixels_sample_count_buffer->resize(m_render_resolution.x * m_render_resolution.y);
	}
	else
	{
		if (m_pixels_squared_luminance_buffer.get_element_count() > 0 || m_pixels_sample_count_buffer->get_element_count() > 0)
			// If one of the buffers isn't freed already, we're going to free it. In this case, we need to synchronize to avoid
			// freeing a buffer that the renderer is actively using in the frame it is rendering right now
			synchronize_kernel();

		m_pixels_squared_luminance_buffer.free();
		m_pixels_sample_count_buffer->free();
	}
}

void GPURenderer::internal_update_restir_di_buffers()
{
	if (m_path_tracer_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		// ReSTIR DI enabled
		bool initial_candidates_reservoir_needs_resize = m_restir_di_state.initial_candidates_reservoirs.get_element_count() == 0;
		bool spatial_output_1_needs_resize = m_restir_di_state.spatial_reuse_output_1.get_element_count() == 0;
		bool spatial_output_2_needs_resize = m_restir_di_state.spatial_reuse_output_2.get_element_count() == 0;

		if (initial_candidates_reservoir_needs_resize || spatial_output_1_needs_resize || spatial_output_2_needs_resize)
			// Synchronizing because we don't want to resize the buffer while the renderer is rendering a frame
			synchronize_kernel();

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
			// If one of the buffers isn't freed already, we're going to free it. In this case, we need to synchronize to avoid
			// freeing a buffer that the renderer is actively using in the frame it is rendering right now
			synchronize_kernel();

		m_restir_di_state.initial_candidates_reservoirs.free();
		m_restir_di_state.spatial_reuse_output_1.free();
		m_restir_di_state.spatial_reuse_output_2.free();
	}
}

void GPURenderer::render()
{
	// Making sure kernels are compiled
	ThreadManager::join_threads(ThreadManager::COMPILE_KERNEL_PASS_THREAD_KEY);

	int tile_size_x = 8;
	int tile_size_y = 8;

	int2 nb_groups = make_int2(std::ceil(m_render_resolution.x / (float)tile_size_x), std::ceil(m_render_resolution.y / (float)tile_size_y));

	// TODO try launch async on the same stream and see performance
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_start_event, m_main_stream));

	for (int i = 1; i <= m_render_data.render_settings.samples_per_frame; i++)
	{
		if (i == m_render_data.render_settings.samples_per_frame)
			// Last sample of the frame so we are going to enable the update 
			// of the status buffers (number of pixels converged, how many rays still
			// active, ...)
			m_render_data.render_settings.do_update_status_buffers = true;

		update_render_data();

		launch_camera_rays();
		launch_ReSTIR_DI();
		launch_path_tracing();

		m_render_data.render_settings.sample_number++;
		m_render_data.render_settings.denoiser_AOV_accumulation_counter++;
	}

	// Recording GPU frame time stop timestamp and computing the frame time
	OROCHI_CHECK_ERROR(oroEventRecord(m_frame_stop_event, m_main_stream));

	GPUKernel::ComputeElapsedTimeCallbackData* elapsed_time_data = new GPUKernel::ComputeElapsedTimeCallbackData;
	elapsed_time_data->start = m_frame_start_event;
	elapsed_time_data->end = m_frame_stop_event;
	elapsed_time_data->elapsed_time_out = &m_ms_time_per_pass["All"];

	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_main_stream, GPUKernel::compute_elapsed_time_callback, elapsed_time_data));

	m_ms_time_per_pass[GPURenderer::CAMERA_RAYS_FUNC_NAME] = m_camera_ray_pass.get_last_execution_time();
	m_ms_time_per_pass[GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME] = m_restir_initial_candidates_pass.get_last_execution_time();
	m_ms_time_per_pass[GPURenderer::RESTIR_DI_TEMPORAL_REUSE_FUNC_NAME] = m_restir_temporal_reuse_pass.get_last_execution_time();
	m_ms_time_per_pass[GPURenderer::RESTIR_DI_SPATIAL_REUSE_FUNC_NAME] = m_restir_spatial_reuse_pass.get_last_execution_time();
	m_ms_time_per_pass[GPURenderer::PATH_TRACING_KERNEL] = m_path_trace_pass.get_last_execution_time();

	// Saving the camera that we used this frame
	m_previous_frame_camera = m_camera;
	m_was_last_frame_low_resolution = m_render_data.render_settings.do_render_low_resolution();
	// We only reset once so after rendering a frame, we're sure that we don't need to reset anymore 
	// so we're setting the flag to false (it will be set to true again if we need to reset the render
	// again)
	m_render_data.render_settings.need_to_reset = false;
	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_render_data.render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = false;
}

void GPURenderer::launch_camera_rays()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution};

	m_render_data.random_seed = m_rng.xorshift32();
	m_camera_ray_pass.launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
}

void GPURenderer::launch_ReSTIR_DI()
{
	void* launch_args[] = { &m_render_data, &m_render_resolution };

	if (m_path_tracer_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		// If ReSTIR DI is enabled

		configure_ReSTIR_DI_initial_pass();
		m_restir_initial_candidates_pass.launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);

		if (m_render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
		{
			configure_ReSTIR_DI_temporal_pass();

			m_restir_temporal_reuse_pass.launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
		}


		if (m_render_data.render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
		{
			for (int spatial_reuse_pass = 0; spatial_reuse_pass < m_render_data.render_settings.restir_di_settings.spatial_pass.number_of_passes; spatial_reuse_pass++)
			{
				configure_ReSTIR_DI_spatial_pass(spatial_reuse_pass);

				m_restir_spatial_reuse_pass.launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
			}
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
	m_path_trace_pass.launch_timed_asynchronous(8, 8, m_render_resolution.x, m_render_resolution.y, launch_args, m_main_stream);
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

	unmap_buffers();

	m_framebuffer->resize(new_width * new_height);
	m_denoised_framebuffer->resize(new_width * new_height);
	m_normals_AOV_buffer->resize(new_width * new_height);
	m_albedo_AOV_buffer->resize(new_width * new_height);

	m_g_buffer.materials.resize(new_width * new_height);
	m_g_buffer.geometric_normals.resize(new_width * new_height);
	m_g_buffer.shading_normals.resize(new_width * new_height);
	m_g_buffer.view_directions.resize(new_width * new_height);
	m_g_buffer.first_hits.resize(new_width * new_height);
	m_g_buffer.cameray_ray_hit.resize(new_width * new_height);
	m_g_buffer.ray_volume_states.resize(new_width * new_height);

	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		m_pixels_sample_count_buffer->resize(new_width * new_height);
		m_pixels_squared_luminance_buffer.resize(new_width * new_height);
	}

	m_restir_di_state.initial_candidates_reservoirs.resize(new_width * new_height);
	m_restir_di_state.spatial_reuse_output_2.resize(new_width * new_height);
	m_restir_di_state.spatial_reuse_output_1.resize(new_width * new_height);
	// Initializing to spatial_reuse_output_1 arbitrarily as it doesn't matter for the first frame since the
	// buffer is going to be empty anyways and no temporal resampling will happen (because this is
	// the buffer that the temporal reuse pass uses as input)
	m_render_data.render_settings.restir_di_settings.restir_output_reservoirs = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();

	m_pixel_active.resize(new_width * new_height);

	// Recomputing the perspective projection matrix since the aspect ratio
	// may have changed
	float new_aspect = (float)new_width / new_height;
	m_camera.projection_matrix = glm::transpose(glm::perspective(m_camera.vertical_fov, new_aspect, m_camera.near_plane, m_camera.far_plane));
}

void GPURenderer::unmap_buffers()
{
	m_framebuffer->unmap();
	m_normals_AOV_buffer->unmap();
	m_albedo_AOV_buffer->unmap();
	m_pixels_sample_count_buffer->unmap();
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

std::shared_ptr<OpenGLInteropBuffer<int>>& GPURenderer::get_pixels_sample_count_buffer()
{
	return m_pixels_sample_count_buffer;
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

std::shared_ptr<GPUKernelCompilerOptions> GPURenderer::get_path_tracer_options()
{
	return m_path_tracer_options;
}

void GPURenderer::recompile_kernels(bool use_cache)
{
	m_camera_ray_pass.compile(m_hiprt_orochi_ctx->hiprt_ctx, m_path_tracer_options, use_cache);
	m_restir_initial_candidates_pass.compile(m_hiprt_orochi_ctx->hiprt_ctx, m_path_tracer_options, use_cache);
	m_restir_temporal_reuse_pass.compile(m_hiprt_orochi_ctx->hiprt_ctx, m_path_tracer_options, use_cache);
	m_restir_spatial_reuse_pass.compile(m_hiprt_orochi_ctx->hiprt_ctx, m_path_tracer_options, use_cache);
	m_path_trace_pass.compile(m_hiprt_orochi_ctx->hiprt_ctx, m_path_tracer_options, use_cache);
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

void GPURenderer::reset()
{
	if (m_render_data.render_settings.accumulate)
	{
		// Only resetting the seed for deterministic rendering if we're accumulating.
		// If we're not accumulating, we want each frame of the render to be different
		// so we don't get into that if block and we don't reset the seed
		m_rng.m_state.seed = 42;

		m_restir_di_state.odd_frame = false;
	}

	m_render_data.render_settings.denoiser_AOV_accumulation_counter = 0;
	m_render_data.render_settings.sample_number = 0;
	m_render_data.render_settings.samples_per_frame = 1;
	m_render_data.render_settings.need_to_reset = true;

	reset_frame_times();
	internal_clear_m_status_buffers();
}

void GPURenderer::update_render_data()
{
	m_render_data.geom = m_hiprt_scene.geometry.m_geometry;

	m_render_data.buffers.pixels = m_framebuffer->map_no_error();
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

	m_render_data.aux_buffers.denoiser_normals = m_normals_AOV_buffer->map_no_error();
	m_render_data.aux_buffers.denoiser_albedo = m_albedo_AOV_buffer->map_no_error();
	if (m_render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		m_render_data.aux_buffers.pixel_sample_count = m_pixels_sample_count_buffer->map_no_error();
		m_render_data.aux_buffers.pixel_squared_luminance = m_pixels_squared_luminance_buffer.get_device_pointer();
	}

	m_render_data.aux_buffers.still_one_ray_active = m_still_one_ray_active_buffer.get_device_pointer();
	m_render_data.aux_buffers.pixel_active = m_pixel_active.get_device_pointer();
	m_render_data.aux_buffers.stop_noise_threshold_converged_count = reinterpret_cast<AtomicType<unsigned int>*>(m_pixels_converged_count_buffer.get_device_pointer());

	m_render_data.g_buffer.materials = m_g_buffer.materials.get_device_pointer();
	m_render_data.g_buffer.geometric_normals = m_g_buffer.geometric_normals.get_device_pointer();
	m_render_data.g_buffer.shading_normals = m_g_buffer.shading_normals.get_device_pointer();
	m_render_data.g_buffer.view_directions = m_g_buffer.view_directions.get_device_pointer();
	m_render_data.g_buffer.first_hits = m_g_buffer.first_hits.get_device_pointer();
	m_render_data.g_buffer.camera_ray_hit = m_g_buffer.cameray_ray_hit.get_device_pointer();
	m_render_data.g_buffer.ray_volume_states = m_g_buffer.ray_volume_states.get_device_pointer();

	// Setting the pointers for use in reset_render() in the camera rays kernel
	if (m_path_tracer_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		m_render_data.aux_buffers.restir_reservoir_buffer_1 = m_restir_di_state.initial_candidates_reservoirs.get_device_pointer();
		m_render_data.aux_buffers.restir_reservoir_buffer_2 = m_restir_di_state.spatial_reuse_output_1.get_device_pointer();
		m_render_data.aux_buffers.restir_reservoir_buffer_3 = m_restir_di_state.spatial_reuse_output_2.get_device_pointer();
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

	m_render_data.current_camera = m_camera.to_hiprt();
	m_render_data.prev_camera = m_previous_frame_camera.to_hiprt();

	m_render_data.random_seed = m_rng.xorshift32();


	// TODO doesn't seem necessary since this method of back also fails
	glm::vec3 t = m_previous_frame_camera.translation;
	m_render_data.prev_camera_position = make_float3(t.x, t.y, t.z);
		
	float a, b, c;

	a = m_render_data.prev_camera.view_projection.m[0][3] + m_render_data.prev_camera.view_projection.m[0][1];
	b = m_render_data.prev_camera.view_projection.m[1][3] + m_render_data.prev_camera.view_projection.m[1][1];
	c = m_render_data.prev_camera.view_projection.m[2][3] + m_render_data.prev_camera.view_projection.m[2][1];
	m_render_data.bottom_plane_normal = hippt::normalize(make_float3(a, b, c));

	a = m_render_data.prev_camera.view_projection.m[0][3] - m_render_data.prev_camera.view_projection.m[0][1];
	b = m_render_data.prev_camera.view_projection.m[1][3] - m_render_data.prev_camera.view_projection.m[1][1];
	c = m_render_data.prev_camera.view_projection.m[2][3] - m_render_data.prev_camera.view_projection.m[2][1];
	m_render_data.top_plane_normal = hippt::normalize(make_float3(a, b, c));

	a = m_render_data.prev_camera.view_projection.m[0][3] + m_render_data.prev_camera.view_projection.m[0][0];
	b = m_render_data.prev_camera.view_projection.m[1][3] + m_render_data.prev_camera.view_projection.m[1][0];
	c = m_render_data.prev_camera.view_projection.m[2][3] + m_render_data.prev_camera.view_projection.m[2][0];
	m_render_data.left_plane_normal = hippt::normalize(make_float3(a, b, c));

	a = m_render_data.prev_camera.view_projection.m[0][3] - m_render_data.prev_camera.view_projection.m[0][0];
	b = m_render_data.prev_camera.view_projection.m[1][3] - m_render_data.prev_camera.view_projection.m[1][0];
	c = m_render_data.prev_camera.view_projection.m[2][3] - m_render_data.prev_camera.view_projection.m[2][0];
	m_render_data.right_plane_normal = hippt::normalize(make_float3(a, b, c));
}

void GPURenderer::set_hiprt_scene_from_scene(const Scene& scene)
{
	HIPRTScene& hiprt_scene = m_hiprt_scene;
	HIPRTGeometry& geometry = hiprt_scene.geometry;
	
	geometry.m_hiprt_ctx = m_hiprt_orochi_ctx->hiprt_ctx;
	geometry.upload_indices(scene.triangle_indices);
	geometry.upload_vertices(scene.vertices_positions);
	geometry.build_bvh();

	hiprt_scene.has_vertex_normals.resize(scene.has_vertex_normals.size());
	hiprt_scene.has_vertex_normals.upload_data(scene.has_vertex_normals.data());

	hiprt_scene.vertex_normals.resize(scene.vertex_normals.size());
	hiprt_scene.vertex_normals.upload_data(scene.vertex_normals.data());

	hiprt_scene.material_indices.resize(scene.material_indices.size());
	hiprt_scene.material_indices.upload_data(scene.material_indices.data());

	hiprt_scene.materials_buffer.resize(scene.materials.size());
	hiprt_scene.materials_buffer.upload_data(scene.materials.data());

	hiprt_scene.emissive_triangles_count = scene.emissive_triangle_indices.size();
	if (hiprt_scene.emissive_triangles_count > 0)
	{
		hiprt_scene.emissive_triangles_indices.resize(scene.emissive_triangle_indices.size());
		hiprt_scene.emissive_triangles_indices.upload_data(scene.emissive_triangle_indices.data());
	}

	hiprt_scene.texcoords_buffer.resize(scene.texcoords.size());
	hiprt_scene.texcoords_buffer.upload_data(scene.texcoords.data());

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

		hiprt_scene.materials_textures.resize(oro_textures.size());
		hiprt_scene.materials_textures.upload_data(oro_textures.data());

		hiprt_scene.textures_dims.resize(scene.textures_dims.size());
		hiprt_scene.textures_dims.upload_data(scene.textures_dims.data());
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

		std::cerr << "Empty envmap set on the GPURenderer..." << std::endl;

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
