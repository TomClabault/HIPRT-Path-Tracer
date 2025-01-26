/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_RENDER_GRAPH_H
#define RENDERER_RENDER_GRAPH_H

#include "Renderer/RenderPasses/RenderPass.h"

#include <memory>
#include <unordered_map>

class GPURenderer;

class RenderGraph : public RenderPass
{
public:
	RenderGraph();
	RenderGraph(GPURenderer* renderer);

	virtual void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}) override;
	virtual void recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}, bool silent = false, bool use_cache = true) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch() override;
	virtual void post_render_update() override;

	virtual void update_render_data() override;
	virtual void reset() override;
	
	virtual void compute_render_times() override;
	virtual void update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics) override;

	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels() override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

	void add_render_pass(std::shared_ptr<RenderPass> render_pass);
	std::shared_ptr<RenderPass> get_render_pass(const std::string& render_pass_name);

private:
	// Launches all the dependencies (recursively) of the given render pass and
	// then launches the given render pass.
	void launch_render_pass_with_dependencies(std::shared_ptr<RenderPass> render_pass);

	// Whether or not launch() has been called on a given render pass this frame.
	// This is used to know whether a render pass has already been launched this frame
	std::unordered_map<std::shared_ptr<RenderPass>, bool> m_render_pass_launched_this_frame_yet;
	// Whether or not launch(), called on a given render pass, returned true this frame
	// 
	// Because calling launch() on a render pass may not *actually* launch the render pass on the GPU
	// (this can happen for example is a render pass is only being launched every N frames. Only
	// one out of N calls to launch() will actually launch the render pass on the GPU), we
	// will need to know when a render pass has effectively been launched because if it hasn't,
	// we can't get the render pass times for this render pass for example
	std::unordered_map<std::shared_ptr<RenderPass>, bool> m_render_pass_effectively_launched_this_frame;

	// Name --> RenderPass
	// The name is actually just render_pass.get_name()
	std::unordered_map<std::string, std::shared_ptr<RenderPass>> m_render_passes;
};

#endif
