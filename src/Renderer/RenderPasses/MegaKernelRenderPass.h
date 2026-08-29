/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef MEGAKERNEL_RENDER_PASS_H
#define MEGAKERNEL_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/RenderPasses/RenderPass.h"

class MegaKernelRenderPass : public RenderPass
{
public:
	static const std::string MEGAKERNEL_RENDER_PASS_NAME;
	static const std::string MEGAKERNEL_KERNEL;

	MegaKernelRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	MegaKernelRenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

	virtual void update_render_data() override {};
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

private:
	int2_t m_render_resolution = make_int2(0, 0);

	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
};

#endif // #ifndef MEGAKERNEL_RENDER_PASS_H
