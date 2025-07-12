/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_RENDER_PASS_H
#define REGIR_RENDER_PASS_H

#include "Renderer/RenderPasses/RenderPass.h"
#include "Renderer/RenderPasses/ReGIRHashGridStorage.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashCellDataSoAHost.h"

class GPURenderer;

class ReGIRRenderPass: public RenderPass
{
public:
	static const std::string REGIR_GRID_PRE_POPULATE;
	static const std::string REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID;
	static const std::string REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID;
	static const std::string REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID;
	static const std::string REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID;
	static const std::string REGIR_PRE_INTEGRATION_KERNEL_ID;
	static const std::string REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID;
	static const std::string REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID;
	static const std::string REGIR_REHASH_KERNEL_ID;
	static const std::string REGIR_SUPERSAMPLING_COPY_KERNEL_ID;

	static const std::string REGIR_RENDER_PASS_NAME;

	/**
	 * This map contains constants that are the name of the main function of the kernels, their entry points.
	 * They are used when compiling the kernels.
	 *
	 * This means that if you define your camera ray kernel main function as:
	 *
	 * GLOBAL_KERNEL_SIGNATURE(void) CameraRays(HIPRTRenderData render_data, int2 res)
	 *
	 * Then KERNEL_FUNCTION_NAMES[CAMERA_RAYS_KERNEL_ID] = "CameraRays"
	 */
	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;

	/**
	 * Same as 'KERNELfUNCTION_NAMES' but for kernel files
	 */
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	ReGIRRenderPass() {}
	ReGIRRenderPass(GPURenderer* renderer);

	virtual void resize(unsigned int new_width, unsigned int new_height) override {};

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}, bool silent = false, bool use_cache = true) override;
	virtual bool pre_render_update(float delta_time) override;


	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	/**
	 * The prepass in ReGIR is used to shoot rays in every directions from the G-Buffer to discover how many grid cells
	 * are going to be needed for the ReGIR grid.
	 */
	void launch_grid_pre_population(HIPRTRenderData& render_data);
	bool rehash(HIPRTRenderData& render_data);

	void launch_grid_fill_temporal_reuse(HIPRTRenderData& render_data, bool primary_hit, bool for_pre_integration, oroStream_t stream);
	void launch_spatial_reuse(HIPRTRenderData& render_data, bool primary_hit, bool for_pre_integration, oroStream_t stream);
	void launch_supersampling_fill(HIPRTRenderData& render_data);
	void launch_supersampling_copy(HIPRTRenderData& render_data);
	void launch_pre_integration(HIPRTRenderData& render_data);
	void launch_pre_integration_internal(HIPRTRenderData& render_data, bool primary_hit, oroStream_t stream = nullptr);
	void launch_rehashing_kernel(HIPRTRenderData& render_data, bool primary_hit, ReGIRHashGridSoADevice& new_hash_grid_soa, ReGIRHashCellDataSoADevice& new_hash_cell_data);

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void update_render_data() override;

	/**
	 * These 2 functions are overriden just to allow a custom handling of the 'frame skip" feature
	 * such that a frame skip of 3 really reflects that the grid fill pass takes 3 times less time
	 */
	virtual void compute_render_times() override;
	virtual void update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics) override;
	virtual float get_full_frame_time() override;

	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used() const override;

	/**
	 * Returns the VRAM used by ReSTIR DI in MB
	 */
	float get_VRAM_usage() const;

	/**
	 * Returns the total number of cells currently used by the hash grid
	 */
	unsigned int get_number_of_cells_alive(bool primary_hit) const;
	unsigned int get_total_number_of_cells_alive(bool primary_hit) const;

	GPURenderer* get_renderer();

	void update_all_cell_alive_count(HIPRTRenderData& render_data);
	float get_alive_cells_ratio(bool primary_hit) const;
	
private:
	unsigned int m_number_of_cells_alive_primary_hits = 0;
	unsigned int m_number_of_cells_alive_secondary_hits = 0;

	Xorshift32Generator m_local_rng = Xorshift32Generator(42);
	OrochiBuffer<unsigned int> m_grid_cells_alive_count_staging_host_pinned_buffer;

	ReGIRHashGridStorage m_hash_grid_storage;

	oroStream_t m_secondary_stream = nullptr;
	oroEvent_t m_oro_event = nullptr;
	oroEvent_t m_event_pre_integration_duration_start = nullptr;
	oroEvent_t m_event_pre_integration_duration_stop = nullptr;
	// Just a flag to make sure that the pre integration pass indeed ran otherwise,
	// if it didn't run, we cannot compute the GPU events elapsed times
	bool m_pre_integration_executed = false;
};

#endif
