/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
	static const std::string RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID;

	static const std::unordered_map<std::string, std::string> KERNEL_FUNCTION_NAMES;
	static const std::unordered_map<std::string, std::string> KERNEL_FILES;

	ReSTIRGIRenderPass();
	ReSTIRGIRenderPass(GPURenderer* renderer);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache) override;
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
	void configure_temporal_reuse_pass(HIPRTRenderData& render_data);
	void launch_temporal_reuse_pass(HIPRTRenderData& render_data);
	void configure_spatial_reuse_pass(HIPRTRenderData& render_data, int spatial_pass_index);
	void launch_spatial_reuse_pass(HIPRTRenderData& render_data);
	void configure_shading_pass(HIPRTRenderData& render_data);
	void launch_shading_pass(HIPRTRenderData& render_data);
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

	virtual bool is_render_pass_used() const override;
	void request_temporal_bufffers_clear();

	/**
	 * Returns the VRAM used by ReSTIR GI in MB
	 */
	float get_VRAM_usage() const;

private:
	// Events for timing the time taken by spatial reuse
	oroEvent_t m_spatial_reuse_time_start;
	oroEvent_t m_spatial_reuse_time_stop;

	OrochiBuffer<ReSTIRGIReservoir> m_initial_candidates_buffer;
	OrochiBuffer<ReSTIRGIReservoir> m_temporal_buffer;
	OrochiBuffer<ReSTIRGIReservoir> m_spatial_buffer;

	OrochiBuffer<unsigned char> m_per_pixel_spatial_reuse_radius;
	OrochiBuffer<unsigned int> m_per_pixel_spatial_reuse_direction_mask_u;
	OrochiBuffer<unsigned long long int> m_per_pixel_spatial_reuse_direction_mask_ull;

	OrochiBuffer<unsigned long long int> m_spatial_reuse_statistics_hit_total;
	OrochiBuffer<unsigned long long int> m_spatial_reuse_statistics_hit_hits;

	ReSTIRGIReservoir* m_last_temporal_output_reservoirs = nullptr;
	ReSTIRGIReservoir* m_last_restir_output_reservoirs = nullptr;

	int m_initial_candidates_generation_seed;
	bool m_temporal_buffer_clear_requested;
};

#endif
