/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_WAVEFRONT_RENDER_PASS_H
#define RENDERER_WAVEFRONT_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/WavefrontDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstddef>
#include <memory>
#include <string>

class WavefrontRenderPass : public RenderPass
{
public:
	static const std::string WAVEFRONT_RENDER_PASS_NAME;
	static const std::string INITIALIZE_PATHS_KERNEL;
	static const std::string ADVANCE_PATHS_KERNEL;

	WavefrontRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	~WavefrontRenderPass() = default;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

private:
	bool resize_staging_buffers();
	void free_staging_buffers();

	int2_t m_render_resolution						   = make_int2(0, 0);
	bool m_staging_buffers_allocated				   = false;
	std::size_t m_allocated_ray_volume_state_byte_size = 0;

	WavefrontDataHost<OrochiBuffer> m_wavefront_data;
	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
	OrochiBuffer<unsigned int> m_zero_host_pinned;
};

#endif // #ifndef RENDERER_WAVEFRONT_RENDER_PASS_H
