/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/RenderPasses/RenderGraph.h"

RenderGraph::RenderGraph() : RenderGraph(nullptr) {}

RenderGraph::RenderGraph(GPURenderer* renderer) : RenderPass(renderer) {}

void RenderGraph::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->compile(hiprt_orochi_ctx, func_name_sets);
}

void RenderGraph::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->recompile(hiprt_orochi_ctx, func_name_sets, silent, use_cache);
}

void RenderGraph::resize(unsigned int new_width, unsigned int new_height)
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->resize(new_width, new_height);
}

bool RenderGraph::pre_render_update(float delta_time)
{
	bool render_data_invalidated = false;
	for (auto& render_pass : m_render_passes)
		render_data_invalidated |= render_pass.second->pre_render_update(delta_time);

	return render_data_invalidated;
}

bool RenderGraph::launch()
{
	// Resetting the state of whether or not the render passes have been launched this frame or not
	for (auto& render_pass : m_render_passes)
	{
		m_render_pass_launched_this_frame_yet[render_pass.first] = false;
		m_render_pass_effectively_launched_this_frame[render_pass.first] = false;
	}

	// Launching all the render passes
	for (auto& render_pass : m_render_passes)
		launch_render_pass_with_dependencies(render_pass.first, render_pass.second);

	return true;
}

void RenderGraph::launch_render_pass_with_dependencies(const std::string& render_pass_name, std::shared_ptr<RenderPass> render_pass)
{
	if (render_pass == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "The render pass \"%s\" wasn't added to the RenderGraph but appears as a dependency of another render pass!", render_pass_name.c_str());

		return;
	}

	// Launching all the dependencies first
	for (const std::string& dependency : render_pass->get_dependencies())
		launch_render_pass_with_dependencies(dependency, m_render_passes[dependency]);

	// Now launching the render pass itself since all dependencies have been launched
	if (m_render_pass_launched_this_frame_yet[render_pass_name] == false)
	{
		bool effectively_launched = m_render_pass_effectively_launched_this_frame[render_pass_name] = render_pass->launch();
		m_render_pass_launched_this_frame_yet[render_pass_name] = true;
		m_render_pass_effectively_launched_this_frame[render_pass_name] = effectively_launched;
	}
}

void RenderGraph::post_render_update()
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->post_render_update();
}

void RenderGraph::update_render_data()
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->update_render_data();
}

void RenderGraph::reset()
{
	for (auto& render_pass : m_render_passes)
		render_pass.second->reset();
}

void RenderGraph::compute_render_times()
{
	for (auto& render_pass : m_render_passes)
		if (m_render_pass_effectively_launched_this_frame[render_pass.first])
			render_pass.second->compute_render_times();
}

void RenderGraph::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	for (auto& render_pass : m_render_passes)
		if (m_render_pass_effectively_launched_this_frame[render_pass.first])
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

void RenderGraph::add_render_pass(const std::string& render_pass_name, std::shared_ptr<RenderPass> render_pass)
{
	m_render_passes[render_pass_name] = render_pass;
}

std::shared_ptr<RenderPass> RenderGraph::get_render_pass(const std::string& render_pass_name)
{
	return m_render_passes[render_pass_name];
}
