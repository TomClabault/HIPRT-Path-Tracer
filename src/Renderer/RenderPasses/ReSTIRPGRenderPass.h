/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_RENDER_PASSES_RESTIR_PG_RENDER_PASS_H
#define RENDERER_RENDER_PASSES_RESTIR_PG_RENDER_PASS_H

#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/PG/ReSTIRPGDistributionSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/PG/ReSTIRPGSplattingSampleSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/PG/ReSTIRPGSufficientStatisticsSoAHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

class ReSTIRPGRenderPass : public RenderPass
{
public:
	static const std::string RESTIR_PG_RENDER_PASS_NAME;
	static const std::string RESTIR_PG_SPLATTING_KERNEL;
	static const std::string RESTIR_PG_FITTING_KERNEL;
	static const std::string RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL;
	static const std::string RESTIR_PG_RESET_HASH_GRID;
	static const std::string RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL;

	static constexpr unsigned int HASH_GRID_INITIAL_CELL_COUNT = 100000;

	ReSTIRPGRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	ReSTIRPGRenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override {};

	virtual bool is_render_pass_used() const override;

private:
	ReSTIRPGSplattingSampleSoAHost<OrochiBuffer> m_splatting_samples_soa_buffer;

	ReSTIRPGDistributionSoAHost<OrochiBuffer> m_hash_grid_distributions_soa_buffer;
	OrochiBuffer<unsigned int> m_hash_grid_checksums_buffer;
	OrochiBuffer<unsigned int> m_grid_cell_alive_buffer;
	OrochiBuffer<unsigned int> m_grid_cell_alive_count_buffer;
	OrochiBuffer<unsigned int> m_grid_cell_alive_list_buffer;

	// Buffers used during the splatting phase to accumulate sample data (expectation phase of the EM algorithm)
	ReSTIRPGSufficientStatisticsSoAHost<OrochiBuffer> m_hash_grid_distributions_sufficient_statistics_soa_buffer;
};

#endif
