/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_RENDER_PASS_H
#define REGIR_RENDER_PASS_H

#include "Renderer/RenderPasses/RenderPass.h"
#include "Renderer/GPUDataStructures/ReGIRGPUData.h"

class GPURenderer;

class ReGIRRenderPass : public RenderPass
{
public:
	static const std::string REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID;
	static const std::string REGIR_SPATIAL_REUSE_KERNEL_ID;

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

	virtual bool launch() override;
	void launch_grid_fill_temporal_reuse();
	void launch_spatial_reuse();

	virtual void post_render_update() override;
	virtual void update_render_data() override;

	virtual void reset() override;
	void reset_representative_points();

	virtual bool is_render_pass_used() const override;

	/**
	 * Returns the VRAM used by ReSTIR DI in MB
	 */
	float get_VRAM_usage() const;

private:
	// Buffer that contains the ReGIR grid. If temporal reuse is enabled,
	// this buffer will contain one more than one grid worth of space to
	// accomodate for the grid of the past frames for temporal reuse
	OrochiBuffer<ReGIRReservoir> m_grid_buffers;
	OrochiBuffer<ReGIRReservoir> m_spatial_reuse_output_grid_buffer;

	// A buffer that contains a point for each grid cell.
	// 
	// The points are not directly contained but rather this contains
	// the index of the pixel whose point needs to be used. So this pixel
	// index should be used to read into the G-Buffer.
	// 
	// That point is guaranteed to be on a valid surface of the scene and can be used as the origin
	// of shadow rays during visibility reuse
	OrochiBuffer<int> m_representative_points_g_buffer_index;
};

#endif
