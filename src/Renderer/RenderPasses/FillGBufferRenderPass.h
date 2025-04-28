/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CAMERA_RAYS_RENDER_PASS_H
#define CAMERA_RAYS_RENDER_PASS_H

#include "HostDeviceCommon/RenderData.h"
#include "Renderer/GPUDataStructures/GBufferGPUData.h"
#include "Renderer/RenderPasses/RenderPass.h"

class GPURenderer;

class FillGBufferRenderPass : public RenderPass
{
public:
	static const std::string FILL_GBUFFER_RENDER_PASS_NAME;
	static const std::string FILL_GBUFFER_KERNEL;

	FillGBufferRenderPass();
	FillGBufferRenderPass(GPURenderer* renderer);

	virtual void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}) override;
	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch() override;
	virtual void post_sample_update() override {}

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override {};

	size_t get_ray_volume_state_byte_size();
	void resize_g_buffer_ray_volume_states();

private:
	int2 m_render_resolution = make_int2(0, 0);

	// G-buffers of the current frame (camera rays hits) and previous frame
	GBufferGPURenderer m_g_buffer;
	GBufferGPURenderer m_g_buffer_prev_frame;

	// Kernel used for retrieving the size of the RayVolumeState structure on the GPU
	std::shared_ptr<GPUKernel> m_ray_volume_state_byte_size_kernel = nullptr;
};

#endif
