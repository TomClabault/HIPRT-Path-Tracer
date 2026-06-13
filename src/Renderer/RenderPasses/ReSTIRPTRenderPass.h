/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_PT_RENDER_PASS_H
#define RESTIR_PT_RENDER_PASS_H

#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/ReSTIRDirectionalSpatialReuseDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/ReSTIRSPMISDataHost.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Renderer/RenderPasses/RenderPass.h"

class GPURenderer;

class ReSTIRPTRenderPass : public MegaKernelRenderPass
{
public:
	static const std::string RESTIR_PT_RENDER_PASS_NAME;
	static const std::string RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID;
	static const std::string RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID;
	static const std::string RESTIR_PT_SHADING_KERNEL_ID;
	static const std::string RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_SORT_KERNEL_ID;

	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	ReSTIRPTRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;
	virtual bool pre_render_update(float delta_time) override;

	/**
	 * This pass computes the optimal reuse radius and reuse directions to use, per-pixel
	 * during the spatial reuse passes
	 *
	 * This is a no-op if not accumulating i.e. this is only available for offline rendering
	 */
	void compute_optimal_spatial_reuse_radii(HIPRTRenderData& render_data);
	void configure_initial_candidates_pass(HIPRTRenderData& render_data);
	void launch_initial_candidates_pass(HIPRTRenderData& render_data);
	void launch_spmis_create_reuse_cells_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options, ReSTIRPTReservoir* input_reservoirs);
	void configure_temporal_reuse_pass(HIPRTRenderData& render_data);
	void launch_temporal_reuse_pass(HIPRTRenderData& render_data);
	void configure_spatial_reuse_pass(HIPRTRenderData& render_data, int spatial_pass_index);
	void launch_spatial_reuse_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);
	void configure_shading_pass(HIPRTRenderData& render_data);
	void launch_shading_pass(HIPRTRenderData& render_data);
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels() override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;
	virtual void compute_render_times() override;

	virtual bool is_render_pass_used() const override;
	void request_temporal_bufffers_clear();

	/**
	 * Returns the VRAM used by ReSTIR PT in MB
	 */
	float get_VRAM_usage() const;

private:
	// Events for timing the time taken by spatial reuse
	bool m_spatial_reuse_events_recorded = false;
	oroEvent_t m_spatial_reuse_time_start;
	oroEvent_t m_spatial_reuse_time_stop;

	bool m_spmis_sorting_events_recorded = false;
	oroEvent_t m_spmis_sorting_time_start;
	oroEvent_t m_spmis_sorting_time_stop;

	OrochiBuffer<ReSTIRPTReservoir> m_initial_candidates_buffer;
	OrochiBuffer<ReSTIRPTReservoir> m_temporal_buffer;
	OrochiBuffer<ReSTIRPTReservoir> m_spatial_buffer;

	ReSTIRDirectionalSpatialReuseDataHost<OrochiBuffer> m_directional_spatial_reuse_data;
	ReSTIRSPMISDataHost<OrochiBuffer> m_spmis_data;

	ReSTIRPTReservoir* m_last_temporal_output_reservoirs = nullptr;
	ReSTIRPTReservoir* m_last_restir_output_reservoirs	 = nullptr;

	int m_initial_candidates_generation_seed;
	bool m_temporal_buffer_clear_requested;
};

#endif
