/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/GPURendererThread.h"

#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

void GPURendererThread::init(GPURenderer* renderer)
{
	// Configuring the render passes
	m_renderer = renderer;
	m_render_graph = RenderGraph(renderer);
	m_render_data = &renderer->m_render_data;
}

void GPURendererThread::start()
{
	m_render_std_thread = std::thread(&GPURendererThread::render_thread_function, this);
	m_render_std_thread.detach();
}

void GPURendererThread::render_thread_function()
{
	OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->m_hiprt_orochi_ctx->orochi_ctx));

	while (true)
	{
		// Wait for the signal to start rendering
		std::unique_lock<std::mutex> lock(m_render_mutex);
		m_render_condition_variable.wait(lock, [this] { return m_frame_requested || m_exit_requested; });
		if (m_exit_requested)
			return;

		// Reset the render requested flag
		m_frame_requested = false;

		// Perform rendering operations here
		render();
		post_frame_update();
	}
}

void GPURendererThread::setup_render_passes()
{
	std::shared_ptr<FillGBufferRenderPass> camera_rays_render_pass = std::make_shared<FillGBufferRenderPass>(m_renderer);

	std::shared_ptr<ReGIRRenderPass> regir_render_pass = std::make_shared<ReGIRRenderPass>(m_renderer);
	regir_render_pass->add_dependency(camera_rays_render_pass);

	std::shared_ptr<ReSTIRDIRenderPass> restir_di_render_pass = std::make_shared<ReSTIRDIRenderPass>(m_renderer);
	restir_di_render_pass->add_dependency(camera_rays_render_pass);
	restir_di_render_pass->add_dependency(regir_render_pass);

	// Note that the megakernel pass will only be used if ReSTIR GI is not used.
	// But we're still adding the render pass to the render graph in case the user
	// switches from ReSTIR GI to classical path tracing at runtime
	std::shared_ptr<MegaKernelRenderPass> megakernel_render_pass = std::make_shared<MegaKernelRenderPass>(m_renderer);
	megakernel_render_pass->add_dependency(camera_rays_render_pass);
	megakernel_render_pass->add_dependency(restir_di_render_pass);

	std::shared_ptr<ReSTIRGIRenderPass> restir_gi_render_pass = std::make_shared<ReSTIRGIRenderPass>(m_renderer);
	restir_gi_render_pass->add_dependency(camera_rays_render_pass);
	restir_gi_render_pass->add_dependency(restir_di_render_pass);

	std::shared_ptr<GMoNRenderPass> gmon_render_pass = std::make_shared<GMoNRenderPass>(m_renderer);
	// GMoN depends on the main path tracing pass which is the megakernel pass or ReSTIR GI, whichever is
	// active
	gmon_render_pass->add_dependency(megakernel_render_pass);
	gmon_render_pass->add_dependency(restir_gi_render_pass);

	m_render_graph.add_render_pass(camera_rays_render_pass);
	m_render_graph.add_render_pass(regir_render_pass);
	m_render_graph.add_render_pass(restir_di_render_pass);
	m_render_graph.add_render_pass(megakernel_render_pass);
	m_render_graph.add_render_pass(restir_gi_render_pass);
	m_render_graph.add_render_pass(gmon_render_pass);

	m_render_graph.compile(m_renderer->m_hiprt_orochi_ctx, m_renderer->m_func_name_sets);

	if (m_renderer->m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE)
		m_renderer->m_nee_plus_plus.compile_finalize_accumulation_kernel(m_renderer->m_hiprt_orochi_ctx, m_renderer->m_func_name_sets);
}

void GPURendererThread::request_frame()
{
	std::lock_guard<std::mutex> lock(m_render_mutex);

	m_currently_rendering = true;
	m_frame_rendered = false;
	m_frame_requested = true;
	m_render_condition_variable.notify_one();
}

void GPURendererThread::request_exit()
{
	std::lock_guard<std::mutex> lock(m_render_mutex);
	
	m_exit_requested = true;
	m_render_condition_variable.notify_one();
}

void GPURendererThread::wait_on_render_completion()
{
	std::unique_lock<std::mutex> lock(m_render_completex_mutex);

	m_render_completed_condition_variable.wait(lock, [this] { return !m_currently_rendering; });
}

void GPURendererThread::pre_render_update(float delta_time, RenderWindow* render_window)
{
	m_renderer->step_animations(delta_time);

	if (m_render_graph.pre_render_compilation_check(m_renderer->m_hiprt_orochi_ctx, m_renderer->m_func_name_sets, true, true))
		// Some kernels have been recompiled, renderer is now dirty
		render_window->set_render_dirty(true);
	m_renderer->m_render_data_buffers_invalidated |= m_render_graph.pre_render_update(delta_time);

	internal_pre_render_update_clear_device_status_buffers();
	internal_pre_render_update_global_stack_buffer();
	internal_pre_render_update_adaptive_sampling_buffers();
	internal_pre_render_update_nee_plus_plus(delta_time);

	m_renderer->update_render_data();

	m_renderer->m_updated = true;
}

void GPURendererThread::internal_pre_render_update_clear_device_status_buffers()
{
	unsigned char false_data = false;
	unsigned int zero_data = 0;
	// Uploading false to reset the flag
	m_renderer->m_status_buffers.still_one_ray_active_buffer.upload_data(&false_data);
	// Resetting the counter of pixels converged to 0
	m_renderer->m_status_buffers.pixels_converged_count_buffer.upload_data(&zero_data);
}

void GPURendererThread::internal_pre_render_update_adaptive_sampling_buffers()
{
	bool buffers_needed = m_render_data->render_settings.has_access_to_adaptive_sampling_buffers();

	if (buffers_needed)
	{
		bool pixels_squared_luminance_needs_resize = m_renderer->m_pixels_squared_luminance_buffer.size() == 0;
		bool pixels_sample_count_needs_resize = m_renderer->m_pixels_sample_count_buffer.size() == 0;
		bool pixels_converged_sample_count_needs_resize = m_renderer->m_pixels_converged_sample_count_buffer->size() == 0;

		if (pixels_squared_luminance_needs_resize || pixels_sample_count_needs_resize || pixels_converged_sample_count_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			m_renderer->m_render_data_buffers_invalidated = true;

		if (pixels_squared_luminance_needs_resize)
			// Only allocating if it isn't already
			m_renderer->m_pixels_squared_luminance_buffer.resize(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);

		if (pixels_sample_count_needs_resize)
			// Only allocating if it isn't already
			m_renderer->m_pixels_sample_count_buffer.resize(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);

		if (pixels_converged_sample_count_needs_resize)
			m_renderer->m_pixels_converged_sample_count_buffer->resize(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);

	}
	else
	{
		if (m_renderer->m_pixels_squared_luminance_buffer.size() > 0 || m_renderer->m_pixels_sample_count_buffer.size() > 0 || m_renderer->m_pixels_converged_sample_count_buffer->size() > 0)
		{
			m_renderer->m_pixels_squared_luminance_buffer.free();
			m_renderer->m_pixels_sample_count_buffer.free();
			m_renderer->m_pixels_converged_sample_count_buffer->free();

			m_renderer->m_render_data_buffers_invalidated = true;
		}
	}
}

void GPURendererThread::internal_pre_render_update_nee_plus_plus(float delta_time)
{
	if (m_renderer->m_global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_FALSE)
	{
		// Not using NEE++, we just need to free the buffers if they weren't already

		if (m_renderer->m_nee_plus_plus.packed_buffer.size() != 0)
		{
			m_renderer->m_nee_plus_plus.packed_buffer.free();
			m_renderer->m_nee_plus_plus.total_shadow_ray_queries.free();
			m_renderer->m_nee_plus_plus.shadow_rays_actually_traced.free();

			m_renderer->m_render_data_buffers_invalidated = true;
		}

		return;
	}

	float3 min_grid_extent_with_envmap, max_grid_extent_with_envmap;
	m_renderer->m_nee_plus_plus.get_grid_extents(m_renderer->m_nee_plus_plus.grid_dimensions_no_envmap, min_grid_extent_with_envmap, max_grid_extent_with_envmap);

	// Adding (2, 2, 2) for envmap NEE++
	m_render_data->nee_plus_plus.grid_dimensions = m_renderer->m_nee_plus_plus.grid_dimensions_no_envmap + make_int3(2, 2, 2);
	m_render_data->nee_plus_plus.grid_min_point = min_grid_extent_with_envmap;
	m_render_data->nee_plus_plus.grid_max_point = max_grid_extent_with_envmap;

	// Allocating / deallocating buffers
	unsigned int matrix_element_count = m_renderer->m_nee_plus_plus.get_visibility_matrix_element_count(m_render_data->nee_plus_plus.grid_dimensions);
	if (m_renderer->m_nee_plus_plus.packed_buffer.size() != matrix_element_count)
	{
		m_renderer->m_nee_plus_plus.packed_buffer.resize(matrix_element_count);
		m_renderer->m_nee_plus_plus.shadow_rays_actually_traced.resize(1);
		m_renderer->m_nee_plus_plus.total_shadow_ray_queries.resize(1);

		m_renderer->m_render_data_buffers_invalidated = true;
	}

	// Clearing the visibility map if this has been asked by the user
	if (m_render_data->nee_plus_plus.reset_visibility_map)
	{
		// Clearing the visibility map by memseting everything to 0
		m_renderer->m_nee_plus_plus.packed_buffer.memset_whole_buffer(0);
		m_renderer->m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(1);
		m_renderer->m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(1);
	}

	if (m_render_data->render_settings.sample_number > m_renderer->m_nee_plus_plus.stop_update_samples)
		// Past a certain number of samples, there isn't really a point to keep updating, the visibility map
		// is probably converged enough that it doesn't make a difference anymore
		m_render_data->nee_plus_plus.update_visibility_map = false;

	m_renderer->m_nee_plus_plus.milliseconds_before_finalizing_accumulation -= delta_time;
	m_renderer->m_nee_plus_plus.milliseconds_before_finalizing_accumulation = hippt::max(0.0f, m_renderer->m_nee_plus_plus.milliseconds_before_finalizing_accumulation); // Clamping for nice display in ImGui (0.0f instead of negative values)
	if (m_renderer->m_nee_plus_plus.milliseconds_before_finalizing_accumulation <= 0.0f && m_render_data->nee_plus_plus.packed_buffers != nullptr)
	{
		m_renderer->m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_TIMER;

		// Because the visibility map data is packed, we can't just use a memcpy() to copy from the accumulation
		// buffers to the visibilit map, we have to use a kernel that the does unpacking-copy
		void* launch_args[] = { &m_render_data->nee_plus_plus };
		m_renderer->m_nee_plus_plus.finalize_accumulation_kernel->launch_asynchronous(256, 1, matrix_element_count, 1, launch_args, m_renderer->get_main_stream());
	}

	m_renderer->m_nee_plus_plus.statistics_refresh_timer -= delta_time;
	if (m_renderer->m_nee_plus_plus.statistics_refresh_timer <= 0.0f && m_render_data->nee_plus_plus.do_update_shadow_rays_traced_statistics)
	{
		m_renderer->m_nee_plus_plus.statistics_refresh_timer = NEEPlusPlusGPUData::STATISTICS_REFRESH_TIMER;

		OROCHI_CHECK_ERROR(oroMemcpy(&m_renderer->m_nee_plus_plus.total_shadow_ray_queries_cpu, m_renderer->m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
		OROCHI_CHECK_ERROR(oroMemcpy(&m_renderer->m_nee_plus_plus.shadow_rays_actually_traced_cpu, m_renderer->m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
	}
}

void GPURendererThread::internal_pre_render_update_global_stack_buffer()
{
	if (m_renderer->needs_global_bvh_stack_buffer())
	{
		bool buffer_needs_update = false;
		// Buffer isn't allocated
		buffer_needs_update |= m_render_data->global_traversal_stack_buffer.stackData == nullptr;
		// Buffer is allocated but the stack size has been changed (through ImGui probably)
		buffer_needs_update |= m_render_data->global_traversal_stack_buffer_size != m_render_data->global_traversal_stack_buffer.stackSize;

		if (buffer_needs_update)
			m_renderer->recreate_global_bvh_stack_buffer();
	}
	else
	{
		if (m_render_data->global_traversal_stack_buffer.stackData != nullptr)
		{
			// Freeing if the buffer already exists
			HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_renderer->m_hiprt_orochi_ctx->hiprt_ctx, m_render_data->global_traversal_stack_buffer));
			m_render_data->global_traversal_stack_buffer.stackData = nullptr;
		}
	}
}

void GPURendererThread::post_sample_update(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	m_render_graph.post_sample_update_async(render_data, compiler_options);

	render_data.render_settings.sample_number++;
	m_render_data->render_settings.sample_number++;

	render_data.render_settings.denoiser_AOV_accumulation_counter++;
	m_render_data->render_settings.denoiser_AOV_accumulation_counter++;

	// We only reset once so after rendering a frame, we're sure that we don't need to reset anymore 
	// so we're setting the flag to false (it will be set to true again if we need to reset the render
	// again)
	render_data.render_settings.need_to_reset = false;
	m_render_data->render_settings.need_to_reset = false;

	render_data.nee_plus_plus.reset_visibility_map = false;
	m_render_data->nee_plus_plus.reset_visibility_map = false;
}

void GPURendererThread::post_frame_update()
{
	// Saving the current frame camera to be the previous camera of the next frame
	m_renderer->m_previous_frame_camera = m_renderer->m_camera;
}

void GPURendererThread::render()
{
	if (!m_renderer->m_updated)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "render() was called on the GPURenderer without update() being called.");
		Utils::debugbreak();

		return;
	}

	// Resetting the update state since we're now rendering a new frame
	m_renderer->m_updated = false;

	// Making sure kernels are compiled
	ThreadManager::join_threads(ThreadManager::COMPILE_KERNELS_THREAD_KEY);

	if (m_debug_trace_kernel.has_been_compiled())
		render_debug_kernel();
	else
		render_path_tracing();
}

void GPURendererThread::render_debug_kernel()
{
	m_frame_rendered = false;

	GPUKernelCompilerOptions compiler_options_copy = m_renderer->m_global_compiler_options->deep_copy();
	// Copying the render data here to avoid race concurrency issues with
	// the asynchronous ImGui UI which may also modifiy the render data
	HIPRTRenderData render_data_copy = m_renderer->get_render_data();

	// Updating the previous and current camera
	render_data_copy.current_camera = m_renderer->m_camera.to_hiprt();
	render_data_copy.prev_camera = m_renderer->m_previous_frame_camera.to_hiprt();

	launch_debug_kernel(render_data_copy);

	// Recording GPU frame time stop timestamp and computing the frame time
	struct CallbackPayload
	{
		bool* frame_rendered;
		bool* currently_rendering;
		std::condition_variable* render_completed_condition_variable;
	};

	CallbackPayload* payload = new CallbackPayload;
	payload->currently_rendering = &m_currently_rendering;
	payload->frame_rendered = &m_frame_rendered;
	payload->render_completed_condition_variable = &m_render_completed_condition_variable;

	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_renderer->get_main_stream(), [](void* payload) 
	{
		CallbackPayload* payload_struct = reinterpret_cast<CallbackPayload*>(payload);
		*payload_struct->frame_rendered = true;
		*payload_struct->currently_rendering = false;
		payload_struct->render_completed_condition_variable->notify_all();

		delete payload_struct;
	}, payload));

	post_sample_update(render_data_copy, compiler_options_copy);
}

GPUKernel& GPURendererThread::get_debug_trace_kernel()
{
	return m_debug_trace_kernel;
}

void GPURendererThread::launch_debug_kernel(HIPRTRenderData& render_data)
{
	void* launch_args[] = { &render_data, &m_renderer->m_render_resolution };

	m_render_data->random_number = m_renderer->m_rng.xorshift32();
	m_debug_trace_kernel.launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
}

void GPURendererThread::render_path_tracing()
{
	m_frame_rendered = false;

	GPUKernelCompilerOptions compiler_options_copy = m_renderer->m_global_compiler_options->deep_copy();
	// Copying the render data here to avoid race concurrency issues with
	// the asynchronous ImGui UI which may also modifiy the render data
	HIPRTRenderData render_data_copy = m_renderer->get_render_data();
	// Updating the previous and current camera
	render_data_copy.current_camera = m_renderer->m_camera.to_hiprt();
	render_data_copy.prev_camera = m_renderer->m_previous_frame_camera.to_hiprt();

	for (int i = 1; i <= m_render_data->render_settings.samples_per_frame; i++)
	{
		if (i == m_render_data->render_settings.samples_per_frame)
			// Last sample of the frame so we are going to enable the update 
			// of the status buffers (number of pixels converged, how many rays still
			// active, ...)
			render_data_copy.render_settings.do_update_status_buffers = true;

		m_render_graph.launch_async(render_data_copy, compiler_options_copy);

		post_sample_update(render_data_copy, compiler_options_copy);
	}

	// Recording GPU frame time stop timestamp and computing the frame time
	struct CallbackPayload
	{
		bool* frame_rendered;
		bool* currently_rendering;
		std::condition_variable* render_completed_condition_variable;
	};

	CallbackPayload* payload = new CallbackPayload;
	payload->currently_rendering = &m_currently_rendering;
	payload->frame_rendered = &m_frame_rendered;
	payload->render_completed_condition_variable = &m_render_completed_condition_variable;

	OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_renderer->get_main_stream(), [](void* payload) 
	{
		CallbackPayload* payload_struct = reinterpret_cast<CallbackPayload*>(payload);
		*payload_struct->frame_rendered = true;
		*payload_struct->currently_rendering = false;
		payload_struct->render_completed_condition_variable->notify_all();

		delete payload_struct;
	}, payload));

	m_renderer->m_was_last_frame_low_resolution = m_render_data->render_settings.do_render_low_resolution();
	// We just rendered a new frame so we're setting this flag to true
	// such that the animated components of the scene are not allowed to step
	// their animations until the render window signals the renderer the the
	// frame has been fully rendered and thus that the animations can step forward
	m_renderer->m_animation_state.can_step_animation = false;
}

RenderGraph& GPURendererThread::get_render_graph()
{
	return m_render_graph;
}

std::shared_ptr<GMoNRenderPass> GPURendererThread::get_gmon_render_pass()
{
	return std::dynamic_pointer_cast<GMoNRenderPass>(m_render_graph.get_render_pass(GMoNRenderPass::GMON_RENDER_PASS_NAME));
}

std::shared_ptr<ReGIRRenderPass> GPURendererThread::get_ReGIR_render_pass()
{
	return std::dynamic_pointer_cast<ReGIRRenderPass>(m_render_graph.get_render_pass(ReGIRRenderPass::REGIR_RENDER_PASS_NAME));
}

std::shared_ptr<ReSTIRDIRenderPass> GPURendererThread::get_ReSTIR_DI_render_pass()
{
	return std::dynamic_pointer_cast<ReSTIRDIRenderPass>(m_render_graph.get_render_pass(ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME));
}

std::shared_ptr<ReSTIRGIRenderPass> GPURendererThread::get_ReSTIR_GI_render_pass()
{
	return std::dynamic_pointer_cast<ReSTIRGIRenderPass>(m_render_graph.get_render_pass(ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME));
}

bool GPURendererThread::frame_render_done()
{
	return m_frame_rendered;
}
