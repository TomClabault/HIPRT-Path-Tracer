/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
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

	FillGBufferRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}) override;
	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {}

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override {};

	/**
	 * Returns the size of the RayVolumeState struct on the GPU.
	 *
	 * Useful when the size of the struct changes because the nested dielectrics
	 * stack size changed but we have no easy way to find out what's the new size
	 * of the struct on the CPU to upload the correct data size.
	 *
	 * There's no easy way to find the new size of the struct on the CPU because
	 * the RayVolumeState struct includes a NestedDielectricsInteriorStack struct whose size
	 * is defined at compilation time. If the nested dielectrics stack size changes
	 * at runtime (possible through ImGui), then we need to recompute the size of
	 * the RayVolumeState structure on the CPU to be able to properly resize the
	 * GPU buffers that use the RayVolumeState (in the GBuffer for example).
	 * However, again, that size is determined at compilation time so we can't
	 * know on the CPU what's going to be the new size. To circumvent that, we
	 * use the fact that shader are recompiled on the GPU and so the shaders know
	 * the new size. This function thus launches a kernel on the GPU to querry
	 * the size of the structure.
	 */
	size_t get_ray_volume_state_byte_size();

	void resize_g_buffer_ray_volume_states();

private:
	int2_t m_render_resolution = make_int2(0, 0);

	// G-buffers of the current frame (camera rays hits) and previous frame
	GBufferGPURenderer m_g_buffer;
	GBufferGPURenderer m_g_buffer_prev_frame;

	// Kernel used for retrieving the size of the RayVolumeState structure on the GPU
	std::shared_ptr<GPUKernel> m_ray_volume_state_byte_size_kernel = nullptr;
	size_t m_ray_volume_state_byte_size							   = sizeof(RayVolumeState);
};

#endif
