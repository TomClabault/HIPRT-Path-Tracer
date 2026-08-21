/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/RenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

RenderPass::RenderPass() {}
RenderPass::RenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options) : RenderPass("Unnamed render pass", renderer, options) {}
RenderPass::RenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: m_renderer(renderer), m_compiler_options(options), m_name(name)
{
}

void RenderPass::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void RenderPass::set_compiler_options(std::shared_ptr<GPUKernelCompilerOptions> options)
{
	m_compiler_options = options;
}

void RenderPass::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	for (auto& name_to_kernel : get_all_kernels())
		ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, m_kernels[name_to_kernel.first],
									hiprt_orochi_ctx, std::ref(func_name_sets));
}

void RenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		// Not recompiling if the render pass is disabled / not being used
		return;

	// The default implementation recompiles all the kernels returned by 'get_all_kernels()'
	for (auto& name_to_kernel : get_all_kernels())
		name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
}

void RenderPass::compute_render_times()
{
	if (!is_render_pass_used(*m_compiler_options))
		// No times to compute if the render pass is disabled / not being used
		return;

	// The default implementation iterates over all kernels and adds their time to the
	// render pass times of the renderer
	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	for (auto& name_to_kernel : get_all_kernels())
		render_pass_times[name_to_kernel.first] = m_kernels[name_to_kernel.first]->compute_execution_time_and_reset_execution_count();
}

void RenderPass::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	if (!is_render_pass_used(*m_compiler_options))
		// No metrics to update if the render pass is disabled / not being used
		return;

	// Add the render pass times computed by 'compute_render_times()' (which was called before
	// 'update_perf_metrics') into the performance metrics computer
	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	float samples_per_frame									  = static_cast<float>(m_renderer->get_render_data().render_settings.samples_per_frame);
	if (samples_per_frame <= 0.0f)
		samples_per_frame = 1.0f;

	for (auto& name_to_kernel : get_all_kernels())
		perf_metrics->add_value(name_to_kernel.first, render_pass_times[name_to_kernel.first] / samples_per_frame);
}

float RenderPass::get_full_frame_time()
{
	float sum = 0.0f;

	for (auto& name_to_kernel : get_all_kernels())
		sum += name_to_kernel.second->get_last_execution_time();

	return sum;
}

std::map<std::string, std::shared_ptr<GPUKernel>> RenderPass::get_all_kernels()
{
	// The default implementation just returns all the kernels.
	// Or an empty map if the render pass isn't being used

	if (!is_render_pass_used(*m_compiler_options))
		return {};
	else
		return m_kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> RenderPass::get_tracing_kernels()
{
	// The default implementation just returns all the kernels (assumes that they are all tracing kernesl).
	return get_all_kernels();
}

bool RenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
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

	for (auto& name_to_kernel : get_all_kernels())
		name_to_kernel.second->set_kernel_name(m_name + "::" + name_to_kernel.first);
}
