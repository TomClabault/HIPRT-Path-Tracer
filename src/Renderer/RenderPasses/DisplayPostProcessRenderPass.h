/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_POST_PROCESS_RENDER_PASS_H
#define DISPLAY_POST_PROCESS_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/RenderPasses/RenderPass.h"

class DisplayPostProcessRenderPass : public RenderPass
{
public:
	static const std::string RENDER_PASS_NAME;
	static const std::string DISPLAY_POST_PROCESS_KERNEL;

	DisplayPostProcessRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	/**
	 * Updates the device-side display settings without requiring a new path-tracing frame.
	 */
	void update_display_post_process_settings();

	/**
	 * Runs the display post-process pass against the currently accumulated buffers.
	 * This is used when a display view or post-process setting changes after rendering converged.
	 */
	bool launch_display_only(HIPRTRenderData& render_data);
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {}

	virtual void update_render_data() override {}
	virtual void reset(bool reset_by_camera_movement) override {}

private:
	bool launch_kernel(HIPRTRenderData& render_data);

	int2_t m_render_resolution = make_int2(0, 0);

	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
};

#endif // #ifndef DISPLAY_POST_PROCESS_RENDER_PASS_H
