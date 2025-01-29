/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_GI_RENDER_PASS_H
#define RESTIR_GI_RENDER_PASS_H

#include "Renderer/RenderPasses/RenderPass.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"

class GPURenderer;

class ReSTIRGIRenderPass : public MegaKernelRenderPass
{
public:
	static const std::string RESTIR_GI_RENDER_PASS_NAME;
	static const std::string RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID;
	static const std::string RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_GI_SPATIAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_GI_SHADING_KERNEL_ID;

	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	ReSTIRGIRenderPass();
	ReSTIRGIRenderPass(GPURenderer* renderer);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache) override;
	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch() override;
	virtual void post_render_update() override;

	virtual void update_render_data() override;
	virtual void reset() override;

	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

	virtual bool is_render_pass_used() const override;

private:
	// Events for timing the time taken by spatial reuse
	oroEvent_t m_spatial_reuse_time_start;
	oroEvent_t m_spatial_reuse_time_stop;

	OrochiBuffer<ReSTIRGIReservoir> m_initial_candidates_buffer;
	OrochiBuffer<ReSTIRGIReservoir> m_temporal_buffer;
	OrochiBuffer<ReSTIRGIReservoir> m_spatial_buffer;
};

#endif
