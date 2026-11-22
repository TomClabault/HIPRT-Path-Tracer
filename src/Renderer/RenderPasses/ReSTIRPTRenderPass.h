/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_PT_RENDER_PASS_H
#define RESTIR_PT_RENDER_PASS_H

#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/ReSTIRPTDirectionalSpatialReuseDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReSTIR/ReSTIRPTSPMISDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

class GPURenderer;

class ReSTIRPTRenderPass : public RenderPass
{
public:
	static const std::string RESTIR_PT_RENDER_PASS_NAME;
	static const std::string RESTIR_PT_INITIAL_CANDIDATES_KERNEL_ID;
	static const std::string RESTIR_PT_TEMPORAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID;
	// Kernels of the union domain spatial reuse (the shadow rays split out of the heavy kernels), see SpatialSPMISUnionDomainShadowSplit.h
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_KERNEL_ID;
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_KERNEL_ID;
	static const std::string RESTIR_PT_SHADING_KERNEL_ID;
	static const std::string RESTIR_PT_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_RESET_BUFFERS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_RESET_CELLS_DATA_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_COUNT_CELLS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_COMPUTE_OFFSETS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_SORT_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_BUILD_CDFS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_COMPUTE_CELLS_VARIANCE_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_BUILD_NEIGHBOR_RECORDS_KERNEL_ID;
	static const std::string RESTIR_PT_SPMIS_BUILD_ALIAS_TABLES_KERNEL_ID;

	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	static constexpr float AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE = 0.025f;

	ReSTIRPTRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;
	virtual bool pre_frame_render_update(float delta_time) override;
	bool pre_render_update_directional_reuse_buffers(const HIPRTRenderData& render_data);
	bool pre_render_update_spmis_buffers(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);
	/**
	 * Allocates the per-pixel state of the union domain spatial reuse kernels when ReSTIRPTSettings::do_spmis_union_domain_reuse() is true and the
	 * spatial reuse pass is enabled, frees it otherwise. Returns true if the buffer was allocated or freed
	 */
	bool pre_render_update_spmis_union_domain_shadow_split_buffer(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);
	void resize_spmis_union_domain_shadow_split_buffer(unsigned int width, unsigned int height);
	/**
	 * Returns true if the buffer was allocated
	 */
	bool free_spmis_union_domain_shadow_split_buffer();

	/**
	 * This pass computes the optimal reuse radius and reuse directions to use, per-pixel
	 * during the spatial reuse passes
	 *
	 * This is a no-op if not accumulating i.e. this is only available for offline rendering
	 */
	void compute_optimal_spatial_reuse_radii(HIPRTRenderData& render_data);
	void configure_initial_candidates_pass(HIPRTRenderData& render_data);
	virtual void launch_initial_candidates_pass(HIPRTRenderData& render_data);
	void launch_spmis_create_reuse_cells_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options, ReSTIRPTReservoir* input_reservoirs);
	void resize_cdf_luts_buffer(HIPRTRenderData& render_data, unsigned int cell_alive_count);
	/**
	 * Number of blocks of 'block_size' threads of the SPMIS cell kernel 'kernel_id' (build CDFs, cells variance, build alias tables): enough to fill the GPU
	 * once, at most 'num_cells'. The blocks loop over the alive cells, whose count is only known on the GPU
	 */
	unsigned int get_spmis_cell_kernel_block_count(const std::string& kernel_id, unsigned int block_size, unsigned int num_cells);
	void configure_temporal_reuse_pass(HIPRTRenderData& render_data);
	void launch_temporal_reuse_pass(HIPRTRenderData& render_data);
	void configure_spatial_reuse_pass(HIPRTRenderData& render_data, int spatial_pass_index);
	/**
	 * Returns true if the spatial reuse of this frame runs the kernels of the SPMIS union domain reuse (see SpatialSPMISUnionDomainShadowSplit.h) instead
	 * of the legacy SPMIS spatial reuse kernel (RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID): ReSTIRPTSettings::do_spmis_union_domain_reuse() is true and the
	 * buffers of the union domain reuse are allocated for the render resolution
	 */
	bool use_spmis_union_domain_kernels(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) const;
	void launch_spmis_union_domain_shadow_split_kernels(HIPRTRenderData& render_data);
	void launch_spatial_reuse_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);
	void configure_shading_pass(HIPRTRenderData& render_data);
	void launch_shading_pass(HIPRTRenderData& render_data);
	void upload_render_data(const std::string& kernel_id, HIPRTRenderData& render_data);
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;
	void request_temporal_bufffers_clear();

	/**
	 * Returns the VRAM used by ReSTIR PT in MB
	 */
	virtual float get_VRAM_usage() const;

protected:
	/**
	 * For the render passes that run the ReSTIR PT passes of this render pass with their own initial candidates
	 */
	ReSTIRPTRenderPass(const std::string& render_pass_name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	/**
	 * The path sampling strategy (PATH_SAMPLING_RESTIR_PT, ...) with which this render pass is used
	 */
	virtual int get_path_sampling_strategy() const;

private:
	OrochiBuffer<ReSTIRPTReservoir> m_initial_candidates_buffer;
	// See ReSTIRPTInitialCandidatesPassSettings::initial_candidates_working_samples_buffer
	OrochiBuffer<ReSTIRPTReservoirSample> m_initial_candidates_working_samples_buffer;
	OrochiBuffer<ReSTIRPTReservoir> m_temporal_buffer;
	OrochiBuffer<ReSTIRPTReservoir> m_spatial_buffer;

	ReSTIRPTDirectionalSpatialReuseDataHost<OrochiBuffer> m_directional_spatial_reuse_data;
	// Number of host pinned render data slots used in turn by upload_render_data(). More than the uploads of one frame so that the wait on the
	// event of a slot practically never blocks
	static constexpr unsigned int RENDER_DATA_UPLOAD_SLOT_COUNT = 32;
	// One host pinned render data per upload of a frame: the asynchronous copy reads its slot when it executes on the GPU, so the slot must not be
	// overwritten by a later upload before that
	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
	// Recorded on the main stream after the copy of each slot. A slot is only rewritten once its event completed
	std::vector<oroEvent_t> m_render_data_upload_events = std::vector<oroEvent_t>(RENDER_DATA_UPLOAD_SLOT_COUNT, nullptr);
	unsigned int m_next_render_data_upload_slot			= 0;
	ReSTIRPTSPMISDataHost<OrochiBuffer> m_spmis_data;
	// Per-pixel state of the union domain spatial reuse kernels (every array of ReSTIRPTSPMISUnionDomainShadowSplitBuffers in one allocation, in
	// elements of 16 bytes so that the element count stays small at large resolutions) and the number of pixels it is allocated for. Only allocated
	// while the union domain reuse runs
	OrochiBuffer<float4_t> m_spmis_union_domain_shadow_split_buffer;
	unsigned int m_spmis_union_domain_shadow_split_pixel_count = 0;

	ReSTIRPTReservoir* m_last_temporal_output_reservoirs  = nullptr;
	ReSTIRPTReservoir* m_last_restir_pt_output_reservoirs = nullptr;

	int m_initial_candidates_generation_seed;
	bool m_temporal_buffer_clear_requested;
};

#endif // #ifndef RESTIR_PT_RENDER_PASS_H
