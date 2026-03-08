/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/GPURendererThread.h"

#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string GPURendererThread::RENDER_GRAPH_FULL_NAME			 = "FullRenderGraph";
const std::string GPURendererThread::RENDER_GRAPH_INTERACTIVITY_NAME = "InteractivityRenderGraph";

void GPURendererThread::init(RenderWindow* render_window, GPURenderer* renderer)
{
	// Configuring the render passes
	m_render_window									 = render_window;
	m_renderer										 = renderer;
	m_render_graphs[RENDER_GRAPH_FULL_NAME]			 = RenderGraph(renderer, std::make_shared<GPUKernelCompilerOptions>());
	m_render_graphs[RENDER_GRAPH_INTERACTIVITY_NAME] = RenderGraph(renderer, std::make_shared<GPUKernelCompilerOptions>());

	m_active_render_graph = &m_render_graphs[RENDER_GRAPH_FULL_NAME];
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

void GPURendererThread::resize(int new_width, int new_height)
{
	for (auto& [rg_name, render_graph] : m_render_graphs)
		render_graph.resize(new_width, new_height);
}

void GPURendererThread::update_is_render_pass_used()
{
	m_active_render_graph->update_is_render_pass_used();
}

void GPURendererThread::reset(bool reset_by_camera_movement)
{
	for (auto& [rg_name, render_graph] : m_render_graphs)
		render_graph.reset(reset_by_camera_movement);
}

void GPURendererThread::setup_render_graphs()
{
	RenderGraph& render_graph_full = m_render_graphs[RENDER_GRAPH_FULL_NAME];

	std::shared_ptr<FillGBufferRenderPass> camera_rays_render_pass	 = render_graph_full.create_render_pass<FillGBufferRenderPass>();
	std::shared_ptr<NEEPlusPlusRenderPass> nee_plus_plus_render_pass = render_graph_full.create_render_pass<NEEPlusPlusRenderPass>();

	std::shared_ptr<ReGIRRenderPass> regir_render_pass = render_graph_full.create_render_pass<ReGIRRenderPass>();
	regir_render_pass->add_dependency(camera_rays_render_pass);
	regir_render_pass->add_dependency(nee_plus_plus_render_pass);

	std::shared_ptr<ReSTIRDIRenderPass> restir_di_render_pass = render_graph_full.create_render_pass<ReSTIRDIRenderPass>();
	restir_di_render_pass->add_dependency(camera_rays_render_pass);
	restir_di_render_pass->add_dependency(regir_render_pass);

	// Note that the megakernel pass will only be used if ReSTIR GI is not used.
	// But we're still adding the render pass to the render graph in case the user
	// switches from ReSTIR GI to classical path tracing at runtime
	std::shared_ptr<MegaKernelRenderPass> megakernel_render_pass = render_graph_full.create_render_pass<MegaKernelRenderPass>();
	megakernel_render_pass->add_dependency(camera_rays_render_pass);
	megakernel_render_pass->add_dependency(restir_di_render_pass);
	megakernel_render_pass->add_dependency(regir_render_pass);

	std::shared_ptr<ReSTIRGIRenderPass> restir_gi_render_pass = render_graph_full.create_render_pass<ReSTIRGIRenderPass>();
	restir_gi_render_pass->add_dependency(camera_rays_render_pass);
	restir_gi_render_pass->add_dependency(restir_di_render_pass);
	restir_gi_render_pass->add_dependency(regir_render_pass);

	std::shared_ptr<GMoNRenderPass> gmon_render_pass = render_graph_full.create_render_pass<GMoNRenderPass>();
	// GMoN depends on the main path tracing pass which
	// is the megakernel pass or ReSTIR GI, whichever is active
	// because we want the values of the samples accumulated in the GMoN sets
	// so far
	gmon_render_pass->add_dependency(megakernel_render_pass);
	gmon_render_pass->add_dependency(restir_gi_render_pass);

	std::shared_ptr<SSBNPermutationRenderPass> ssbn_permutation_render_pass = render_graph_full.create_render_pass<SSBNPermutationRenderPass>();
	ssbn_permutation_render_pass->add_dependency(megakernel_render_pass);
	ssbn_permutation_render_pass->add_dependency(restir_gi_render_pass);

	render_graph_full.add_render_pass(camera_rays_render_pass);
	render_graph_full.add_render_pass(nee_plus_plus_render_pass);
	render_graph_full.add_render_pass(regir_render_pass);
	render_graph_full.add_render_pass(restir_di_render_pass);
	render_graph_full.add_render_pass(megakernel_render_pass);
	render_graph_full.add_render_pass(restir_gi_render_pass);
	render_graph_full.add_render_pass(gmon_render_pass);
	render_graph_full.add_render_pass(ssbn_permutation_render_pass);

	render_graph_full.compile(m_renderer->m_hiprt_orochi_ctx, m_renderer->m_func_name_sets);

	render_graph_full.set_render_window(m_render_window);

	/**
	 * Render graph interactivity
	 */

	RenderGraph& render_graph_interactivity = m_render_graphs[RENDER_GRAPH_INTERACTIVITY_NAME];
	render_graph_interactivity.get_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_BASE_LIGHT_TREE_SG);
	render_graph_interactivity.get_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PSS_BSDF);

	std::shared_ptr<FillGBufferRenderPass> camera_rays_render_pass_interactivity = render_graph_interactivity.create_render_pass<FillGBufferRenderPass>();
	std::shared_ptr<MegaKernelRenderPass> megakernel_render_pass_interactivity	 = render_graph_interactivity.create_render_pass<MegaKernelRenderPass>();
	megakernel_render_pass_interactivity->add_dependency(camera_rays_render_pass_interactivity);

	render_graph_interactivity.add_render_pass(camera_rays_render_pass_interactivity);
	render_graph_interactivity.add_render_pass(megakernel_render_pass_interactivity);

	render_graph_interactivity.set_render_window(m_render_window);
	render_graph_interactivity.compile(m_renderer->m_hiprt_orochi_ctx, m_renderer->m_func_name_sets);
}

void GPURendererThread::request_frame(HIPRTRenderData& render_data_for_frame, GPUKernelCompilerOptions& compiler_options_for_frame)
{
	std::lock_guard<std::mutex> lock(m_render_mutex);

	m_render_data_for_frame		 = render_data_for_frame;
	m_compiler_options_for_frame = compiler_options_for_frame.deep_copy();

	m_currently_rendering = true;
	m_frame_rendered	  = false;
	m_frame_requested	  = true;
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

void GPURendererThread::pre_render_update(float delta_time)
{
	m_renderer->step_animations(delta_time);
	m_renderer->prepare_light_sampling_data_structures();

	m_renderer->update_render_data();

	if (m_active_render_graph->pre_render_compilation_check(m_renderer->m_hiprt_orochi_ctx, m_renderer->m_func_name_sets, true, true))
		// Some kernels have been recompiled, renderer is now dirty
		m_render_window->set_render_dirty(true);
	m_renderer->m_render_data_buffers_invalidated |= m_active_render_graph->pre_render_update(delta_time);

	internal_pre_render_update_clear_device_status_buffers();
	internal_pre_render_update_global_stack_buffer();
	internal_pre_render_update_adaptive_sampling_buffers();

	m_active_render_graph->update_render_data();

	m_renderer->m_updated = true;
}

void GPURendererThread::internal_pre_render_update_clear_device_status_buffers()
{
	unsigned char false_data = false;
	unsigned int zero_data	 = 0;
	// Uploading false to reset the flag
	m_renderer->m_status_buffers.still_one_ray_active_buffer.upload_data(&false_data);
	// Resetting the counter of pixels converged to 0
	m_renderer->m_status_buffers.pixels_converged_count_buffer.upload_data(&zero_data);
}

void GPURendererThread::internal_pre_render_update_adaptive_sampling_buffers()
{
	bool buffers_needed = m_renderer->get_render_data().render_settings.has_access_to_adaptive_sampling_buffers();

	if (buffers_needed)
	{
		bool pixels_squared_luminance_needs_resize		= m_renderer->m_pixels_squared_luminance_buffer.size() == 0;
		bool pixels_sample_count_needs_resize			= m_renderer->m_pixels_sample_count_buffer.size() == 0;
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
		if (m_renderer->m_pixels_squared_luminance_buffer.size() > 0 || m_renderer->m_pixels_sample_count_buffer.size() > 0 ||
			m_renderer->m_pixels_converged_sample_count_buffer->size() > 0)
		{
			m_renderer->m_pixels_squared_luminance_buffer.free();
			m_renderer->m_pixels_sample_count_buffer.free();
			m_renderer->m_pixels_converged_sample_count_buffer->free();

			m_renderer->m_render_data_buffers_invalidated = true;
		}
	}
}

void GPURendererThread::internal_pre_render_update_global_stack_buffer()
{
	if (m_renderer->needs_global_bvh_stack_buffer())
	{
		bool buffer_needs_update = false;
		// Buffer isn't allocated
		buffer_needs_update |= m_renderer->get_render_data().global_traversal_stack_buffer.stackData == nullptr;
		// Buffer is allocated but the stack size has been changed (through ImGui probably)
		buffer_needs_update |= m_renderer->get_render_data().global_traversal_stack_buffer_size !=
							   m_renderer->get_render_data().global_traversal_stack_buffer.stackSize;

		if (buffer_needs_update)
			m_renderer->recreate_global_bvh_stack_buffer();
	}
	else
	{
		if (m_renderer->get_render_data().global_traversal_stack_buffer.stackData != nullptr)
		{
			// Freeing if the buffer already exists
			HIPRT_CHECK_ERROR(hiprtDestroyGlobalStackBuffer(m_renderer->m_hiprt_orochi_ctx->hiprt_ctx,
															m_renderer->get_render_data().global_traversal_stack_buffer));
			m_renderer->get_render_data().global_traversal_stack_buffer.stackData = nullptr;
		}
	}
}

void GPURendererThread::post_sample_update(HIPRTRenderData& render_data_for_frame, GPUKernelCompilerOptions& compiler_options)
{
	// This function also updates render_data_for_frame such that if multiple samples are dispatched per frame then the next samples are going to get the
	// updated render data as well

	m_active_render_graph->post_sample_update_async(render_data_for_frame, compiler_options);

	render_data_for_frame.render_settings.sample_number++;
	m_renderer->get_render_data().render_settings.sample_number++;

	render_data_for_frame.render_settings.denoiser_AOV_accumulation_counter++;
	m_renderer->get_render_data().render_settings.denoiser_AOV_accumulation_counter++;

	// We only reset once so after rendering a frame, we're sure that we don't need to reset anymore
	// so we're setting the flag to false (it will be set to true again if we need to reset the render
	// again)
	render_data_for_frame.render_settings.need_to_reset = false;
	m_renderer->get_render_data().render_settings.need_to_reset = false;

	render_data_for_frame.render_settings.need_to_reset_random_seeds = false;
	m_renderer->get_render_data().render_settings.need_to_reset_random_seeds = false;

	render_data_for_frame.nee_plus_plus.m_reset_visibility_map		   = false;
	m_renderer->get_render_data().nee_plus_plus.m_reset_visibility_map = false;
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
		Debug::debugbreak();

		return;
	}

	// Resetting the update state since we're now rendering a new frame
	m_renderer->m_updated = false;

	// Making sure kernels are compiled
	ThreadManager::join_threads(ThreadManager::COMPILE_KERNELS_THREAD_KEY);

	render_internal();
}

void GPURendererThread::render_internal()
{
	m_frame_rendered = false;

	// Updating the previous and current camera
	m_render_data_for_frame.current_camera = m_renderer->m_camera.to_hiprt(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);
	m_render_data_for_frame.prev_camera	   = m_renderer->m_previous_frame_camera.to_hiprt(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);

	for (int i = 1; i <= m_render_data_for_frame.render_settings.samples_per_frame; i++)
	{
		if (i == m_render_data_for_frame.render_settings.samples_per_frame)
			// Last sample of the frame so we are going to enable the update
			// of the status buffers (number of pixels converged, how many rays still
			// active, ...)
			m_render_data_for_frame.render_settings.do_update_status_buffers = true;

		m_active_render_graph->launch_async(m_render_data_for_frame, m_compiler_options_for_frame);

		post_sample_update(m_render_data_for_frame, m_compiler_options_for_frame);
	}

	// Recording GPU frame time stop timestamp and computing the frame time
	struct CallbackPayload
	{
		bool* frame_rendered;
		bool* currently_rendering;
		std::condition_variable* render_completed_condition_variable;
	};

	CallbackPayload* payload					 = new CallbackPayload;
	payload->currently_rendering				 = &m_currently_rendering;
	payload->frame_rendered						 = &m_frame_rendered;
	payload->render_completed_condition_variable = &m_render_completed_condition_variable;

	OROCHI_CHECK_ERROR(oroLaunchHostFunc(
							m_renderer->get_main_stream(),
							[](void* payload)
							{
								CallbackPayload* payload_struct		 = reinterpret_cast<CallbackPayload*>(payload);
								*payload_struct->frame_rendered		 = true;
								*payload_struct->currently_rendering = false;
								payload_struct->render_completed_condition_variable->notify_all();

								delete payload_struct;
							},
							payload));

	m_renderer->m_was_last_frame_low_resolution = m_renderer->get_render_data().render_settings.do_render_low_resolution();
	// We just rendered a new frame so we're setting this flag to true
	// such that the animated components of the scene are not allowed to step
	// their animations until the render window signals the renderer the the
	// frame has been fully rendered and thus that the animations can step forward
	m_renderer->m_animation_state.can_step_animation = false;
}

void GPURendererThread::set_active_render_graph(RenderGraph& graph)
{
	m_active_render_graph = &graph;
}

RenderGraph& GPURendererThread::get_active_render_graph()
{
	return *m_active_render_graph;
}

std::unordered_map<std::string, RenderGraph>& GPURendererThread::get_render_graphs()
{
	return m_render_graphs;
}

std::shared_ptr<GMoNRenderPass> GPURendererThread::get_gmon_render_pass()
{
	return std::dynamic_pointer_cast<GMoNRenderPass>(m_active_render_graph->get_render_pass(GMoNRenderPass::GMON_RENDER_PASS_NAME));
}

std::shared_ptr<GMoNRenderPass> GPURendererThread::get_gmon_render_pass() const
{
	return std::dynamic_pointer_cast<GMoNRenderPass>(m_active_render_graph->get_render_pass(GMoNRenderPass::GMON_RENDER_PASS_NAME));
}

std::shared_ptr<ReGIRRenderPass> GPURendererThread::get_ReGIR_render_pass()
{
	return std::dynamic_pointer_cast<ReGIRRenderPass>(m_active_render_graph->get_render_pass(ReGIRRenderPass::REGIR_RENDER_PASS_NAME));
}

std::shared_ptr<ReSTIRDIRenderPass> GPURendererThread::get_ReSTIR_DI_render_pass()
{
	return std::dynamic_pointer_cast<ReSTIRDIRenderPass>(m_active_render_graph->get_render_pass(ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME));
}

std::shared_ptr<ReSTIRGIRenderPass> GPURendererThread::get_ReSTIR_GI_render_pass()
{
	return std::dynamic_pointer_cast<ReSTIRGIRenderPass>(m_active_render_graph->get_render_pass(ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME));
}

std::shared_ptr<NEEPlusPlusRenderPass> GPURendererThread::get_NEE_plus_plus_render_pass()
{
	return std::dynamic_pointer_cast<NEEPlusPlusRenderPass>(m_active_render_graph->get_render_pass(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME));
}

bool GPURendererThread::frame_render_done()
{
	return m_frame_rendered;
}
