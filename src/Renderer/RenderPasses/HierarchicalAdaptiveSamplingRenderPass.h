/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_RENDER_PASSES_HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_PASS_H
#define RENDERER_RENDER_PASSES_HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HierarchicalAdaptiveSampling.h"
#include "Renderer/RenderPasses/RenderPass.h"

class HierarchicalAdaptiveSamplingRenderPass : public RenderPass
{
public:
	static const std::string RENDER_PASS_NAME;
	static const std::string COMPUTE_ERROR_KERNEL;
	static const std::string BUILD_ROWS_KERNEL;
	static const std::string BUILD_COLUMNS_KERNEL;
	static const std::string BUILD_HIERARCHY_KERNEL;
	static const std::string RESOLVE_MASK_KERNEL;

	HierarchicalAdaptiveSamplingRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets = {},
											  bool silent										  = false,
											  bool use_cache									  = true) override;
	virtual void recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
						   const std::vector<hiprtFuncNameSet>& func_name_sets = {},
						   bool silent										   = false,
						   bool use_cache									   = true) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {}
	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;
	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

private:
	void configure_kernel(const std::string& kernel_id, const std::string& function_name, const std::string& kernel_file_name);
	void upload_render_data(const std::string& kernel_id, HIPRTRenderData& render_data);
	void free_buffers();

	int2_t m_render_resolution = make_int2(0, 0);

	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
	OrochiBuffer<unsigned int> m_build_depths_host_pinned;
	OrochiBuffer<float> m_error;
	OrochiBuffer<float> m_summed_area;
	OrochiBuffer<HierarchicalAdaptiveSamplingNode> m_nodes;
	OrochiBuffer<unsigned int> m_node_count;
	OrochiBuffer<unsigned int> m_level_node_count;
};

#endif // #ifndef RENDERER_RENDER_PASSES_HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_PASS_H
