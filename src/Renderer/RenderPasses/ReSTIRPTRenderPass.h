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
	static const std::string RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_KERNEL_ID;
	// Kernels of the shadow split version of the union domain spatial reuse, see SpatialSPMISUnionDomainShadowSplit.h
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
	 * Allocates the per-pixel state of the shadow split union domain kernels when ReSTIRPTSettings::use_spmis_union_domain_shadow_split_kernels() is true,
	 * frees it otherwise. Returns true if the buffer was allocated or freed
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
	void launch_initial_candidates_pass(HIPRTRenderData& render_data);
	void launch_spmis_create_reuse_cells_pass(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options, ReSTIRPTReservoir* input_reservoirs);
	void resize_cdf_luts_buffer(HIPRTRenderData& render_data, unsigned int cell_alive_count);
	void configure_temporal_reuse_pass(HIPRTRenderData& render_data);
	void launch_temporal_reuse_pass(HIPRTRenderData& render_data);
	void configure_spatial_reuse_pass(HIPRTRenderData& render_data, int spatial_pass_index);
	/**
	 * Returns true if the spatial reuse of this frame runs the SPMIS union domain reuse kernel (RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_KERNEL_ID)
	 * instead of the legacy SPMIS spatial reuse kernel (RESTIR_PT_SPATIAL_REUSE_SPMIS_KERNEL_ID)
	 */
	bool use_spmis_union_domain_kernel(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) const;
	/**
	 * Returns true if the union domain reuse of this frame runs as the shadow split kernels (see SpatialSPMISUnionDomainShadowSplit.h) instead of the single
	 * union domain kernel: the union domain kernel is used, ReSTIRPTSettings::use_spmis_union_domain_shadow_split_kernels() is true and the state of the shadow
	 * split kernels is allocated for the render resolution
	 */
	bool use_spmis_union_domain_shadow_split_kernels(const HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) const;
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
	float get_VRAM_usage() const;

private:
	/**
	 * Reserves 'element_count' elements of type T at a 256 byte aligned offset of the buffer of the shadow split union domain kernels whose first
	 * 'buffer_byte_size' bytes are already reserved, and moves 'buffer_byte_size' past them. 'out_array' points to the reserved elements in the buffer
	 * 'buffer_device_pointer', or is nullptr when the buffer pointer is nullptr (only the size of the buffer is computed)
	 */
	template <typename T>
	static void reserve_spmis_union_domain_shadow_split_array(unsigned char* buffer_device_pointer,
															  size_t& buffer_byte_size,
															  size_t element_count,
															  T*& out_array);

	/**
	 * Layout of all the arrays of ReSTIRPTSPMISUnionDomainShadowSplitBuffers in one buffer for 'pixel_count' pixels (460 bytes per pixel with the
	 * maximum pool size and probe count of ReSTIRPTOptions.h, plus alignment). Sets the pointers of 'shadow_split_buffers' into the buffer
	 * 'buffer_device_pointer' (nullptr pointers when 'buffer_device_pointer' is nullptr) and returns the byte size of the buffer
	 */
	static size_t layout_spmis_union_domain_shadow_split_buffers(unsigned char* buffer_device_pointer,
																 unsigned int pixel_count,
																 ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers);

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
	// Per-pixel state of the shadow split union domain kernels (every array of ReSTIRPTSPMISUnionDomainShadowSplitBuffers in one allocation, in
	// elements of 16 bytes so that the element count stays small at large resolutions) and the number of pixels it is allocated for. Only allocated
	// while the shadow split kernels run
	OrochiBuffer<float4_t> m_spmis_union_domain_shadow_split_buffer;
	unsigned int m_spmis_union_domain_shadow_split_pixel_count = 0;

	ReSTIRPTReservoir* m_last_temporal_output_reservoirs  = nullptr;
	ReSTIRPTReservoir* m_last_restir_pt_output_reservoirs = nullptr;

	int m_initial_candidates_generation_seed;
	bool m_temporal_buffer_clear_requested;
};

#endif // #ifndef RESTIR_PT_RENDER_PASS_H
