/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/RenderGraph.h"

RenderGraph::RenderGraph() : RenderGraph(nullptr) {}

RenderGraph::RenderGraph(GPURenderer* renderer) : RenderPass(renderer) {}

void RenderGraph::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	for (auto& name_to_render_pass : m_render_passes)
		name_to_render_pass.second->compile(hiprt_orochi_ctx, func_name_sets);
}

void RenderGraph::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	for (auto& name_to_render_pass : m_render_passes)
		name_to_render_pass.second->recompile(hiprt_orochi_ctx, func_name_sets, silent, use_cache);
}

void RenderGraph::resize(unsigned int new_width, unsigned int new_height)
{
	for (auto& name_to_render_pass : m_render_passes)
		name_to_render_pass.second->resize(new_width, new_height);
}

bool RenderGraph::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	m_renderer->synchronize_all_kernels();

	bool recompiled = false;
	m_renderer->take_kernel_compilation_priority();
	for (auto& name_to_render_pass : m_render_passes)
		recompiled |= name_to_render_pass.second->pre_render_compilation_check(hiprt_orochi_ctx, func_name_sets, silent, use_cache);
	m_renderer->release_kernel_compilation_priority();

	return recompiled;
}

void RenderGraph::prepass()
{
	for (auto& name_to_render_pass : m_render_passes)
		name_to_render_pass.second->prepass();
}

bool RenderGraph::pre_render_update(float delta_time)
{
	bool render_data_invalidated = false;
	for (auto& name_to_render_pass : m_render_passes)
		render_data_invalidated |= name_to_render_pass.second->pre_render_update(delta_time);

	// pre_render_update means that this is a new frame
	m_new_frame = true;

	return render_data_invalidated;
}

bool RenderGraph::launch(HIPRTRenderData& render_data)
{
		// Resetting the state of whether or not the render passes have been launched this frame or not
	for (auto& name_to_render_pass : m_render_passes)
	{
		m_render_pass_launched_this_frame_yet[name_to_render_pass.second.get()] = false;

		if (m_new_frame)
			m_render_pass_effectively_launched_this_frame[name_to_render_pass.second.get()] = false;
	}

	// Launching all the render passes
	for (auto& name_to_render_pass : m_render_passes)
		launch_render_pass_with_dependencies(name_to_render_pass.second, render_data);

	// This is not a fresh frame anymore
	m_new_frame = false;

	return true;
}

void RenderGraph::launch_render_pass_with_dependencies(std::shared_ptr<RenderPass> render_pass, HIPRTRenderData& render_data)
{
	if (render_pass == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "The render pass \"%s\" wasn't added to the RenderGraph but appears as a dependency of another render pass!", render_pass->get_name().c_str());

		return;
	}

	if (m_render_pass_launched_this_frame_yet[render_pass.get()] == true)
		// This pas has already been launched
		return;

	// Launching all the dependencies first
	for (std::shared_ptr<RenderPass> dependency : render_pass->get_dependencies())
		launch_render_pass_with_dependencies(dependency, render_data);

	// Now launching the render pass itself since all dependencies have been launched
	bool effectively_launched = render_pass->launch(render_data);
	m_render_pass_launched_this_frame_yet[render_pass.get()] = true;

	if (effectively_launched)
		// Only setting the effectively launched to true if the render pass was launched
		// Otherwise, this leaves it at its current value
		m_render_pass_effectively_launched_this_frame[render_pass.get()] = true;
}

void RenderGraph::post_sample_update(HIPRTRenderData& render_data)
{
	for (auto& name_to_render_pass : m_render_passes)
		name_to_render_pass.second->post_sample_update(render_data);
}

void RenderGraph::update_render_data()
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->update_render_data();
}

void RenderGraph::reset(bool reset_by_camera_movement)
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->reset(reset_by_camera_movement);
}

void RenderGraph::compute_render_times()
{
	for (auto& render_pass : m_render_passes)
		if (m_render_pass_effectively_launched_this_frame[render_pass.second.get()])
			render_pass.second->compute_render_times();
}

void RenderGraph::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	for (auto& render_pass : m_render_passes)
		if (m_render_pass_effectively_launched_this_frame[render_pass.second.get()])
			render_pass.second->update_perf_metrics(perf_metrics);
}

std::map<std::string, std::shared_ptr<GPUKernel>> RenderGraph::get_all_kernels() 
{
	std::map<std::string, std::shared_ptr<GPUKernel>> out;

	// For all render passes
	for (auto& render_pass : m_render_passes)
		// For all the kernels of this render pass
		for (auto& name_to_kernel : render_pass.second->get_all_kernels())
			out[name_to_kernel.first] = name_to_kernel.second;

	return out;
}

std::map<std::string, std::shared_ptr<GPUKernel>> RenderGraph::get_tracing_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> out;

	// For all render passes
	for (auto& render_pass : m_render_passes)
		// For all the kernels of this render pass
		for (auto& name_to_kernel : render_pass.second->get_tracing_kernels())
			out[name_to_kernel.first] = name_to_kernel.second;

	return out;
}

void RenderGraph::add_render_pass(std::shared_ptr<RenderPass> render_pass)
{
	if (m_render_passes.find(render_pass->get_name()) != m_render_passes.end())
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "A render pass with name %s already exists in the render graph. This call to add_render_pass() didn't change anything.", render_pass->get_name().c_str());

		return;
	}

	m_render_passes[render_pass->get_name()] = render_pass;
}

std::shared_ptr<RenderPass> RenderGraph::get_render_pass(const std::string& render_pass_name)
{
	return m_render_passes[render_pass_name];
}

std::unordered_map<std::string, std::shared_ptr<RenderPass>> RenderGraph::get_render_passes()
{
	return m_render_passes;
}
