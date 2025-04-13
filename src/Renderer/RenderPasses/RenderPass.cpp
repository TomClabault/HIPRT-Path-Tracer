/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/RenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

RenderPass::RenderPass() {}
RenderPass::RenderPass(GPURenderer* renderer) : RenderPass(renderer, "Unnamed render pass") {}
RenderPass::RenderPass(GPURenderer* renderer, const std::string& name) : m_renderer(renderer), m_render_data(&m_renderer->get_render_data()), m_name(name) {}

void RenderPass::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	if (!is_render_pass_used())
		return;

	for (auto& name_to_kernel : get_all_kernels())
		ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, m_kernels[name_to_kernel.first], hiprt_orochi_ctx, std::ref(func_name_sets));
}

void RenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		// Not recompiling if the render pass is disabled / not being used
		return;

	// The default implementation recompiles all the kernels returned by 'get_all_kernels()'
	for (auto& name_to_kernel : get_all_kernels())
		name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
}

void RenderPass::compute_render_times()
{
	if (!is_render_pass_used())
		// No times to compute if the render pass is disabled / not being used
		return;

	// The default implementation iterates over all kernels and adds their time to the
	// render pass times of the renderer
	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	for (auto& name_to_kernel : get_all_kernels())
		render_pass_times[name_to_kernel.first] = m_kernels[name_to_kernel.first]->get_last_execution_time();
}

void RenderPass::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	if (!is_render_pass_used())
		// No metrics to update if the render pass is disabled / not being used
		return;

	// Add the render pass times computed by 'compute_render_times()' (which was called before
	// 'update_perf_metrics') into the performance metrics computer
	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	for (auto& name_to_kernel : get_all_kernels())
		perf_metrics->add_value(name_to_kernel.first, render_pass_times[name_to_kernel.first]);
}

std::map<std::string, std::shared_ptr<GPUKernel>> RenderPass::get_all_kernels()
{
	// The default implementation just returns all the kernels.
		// Or an empty map if the render pass isn't being used

	if (!is_render_pass_used())
		return {};
	else
		return m_kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> RenderPass::get_tracing_kernels()
{
	// The default implementation just returns all the kernels (assumes that they are all tracing kernesl).
	return get_all_kernels();
}

bool RenderPass::is_render_pass_used() const
{
	return true;
}

void RenderPass::add_dependency(std::shared_ptr<RenderPass> dependency)
{
	m_dependencies.push_back(dependency);
}

std::vector<std::shared_ptr<RenderPass>>& RenderPass::get_dependencies()
{
	return m_dependencies;
}

const std::string& RenderPass::get_name()
{
	return m_name;
}

void RenderPass::set_name(const std::string& new_name)
{
	m_name = new_name;
}
