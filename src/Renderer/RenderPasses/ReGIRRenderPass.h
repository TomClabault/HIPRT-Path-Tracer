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
	static const std::string REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID;
	static const std::string REGIR_SPATIAL_REUSE_KERNEL_ID;
	static const std::string REGIR_REHASH_KERNEL_ID;

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
	void launch_grid_fill_temporal_reuse(HIPRTRenderData& render_data);
	void launch_spatial_reuse(HIPRTRenderData& render_data);
	void launch_rehashing_kernel(HIPRTRenderData& render_data, ReGIRHashGridSoADevice& new_hash_grid, ReGIRHashCellDataSoADevice& new_hash_cell_data);

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void update_render_data() override;

	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used() const override;

	/**
	 * Returns the VRAM used by ReSTIR DI in MB
	 */
	float get_VRAM_usage() const;

	/**
	 * Returns the total number of cells currently used by the hash grid
	 */
	unsigned int get_number_of_cells() const;
	unsigned int get_number_of_cells_alive() const;

	unsigned int update_cell_alive_count();
	float get_alive_cells_ratio() const;

	ReGIRHashGridStorage m_hash_grid_storage;
	
private:
	int m_current_grid_index = 0;


	unsigned int m_number_of_cells_alive = 0;

	OrochiBuffer<unsigned int> m_grid_cells_alive_count_staging_host_pinned_buffer;
};

#endif
