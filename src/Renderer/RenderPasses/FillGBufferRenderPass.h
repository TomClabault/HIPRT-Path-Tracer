/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CAMERA_RAYS_RENDER_PASS_H
#define CAMERA_RAYS_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
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

	/**
	 * Makes this render pass fill the G-buffers of 'g_buffers_owner' instead of allocating its own.
	 *
	 * Only one render graph renders at a time so the passes of two render graphs can fill the same G-buffers: a second set of
	 * G-buffers would only take VRAM. The owner keeps allocating, resizing and freeing them.
	 *
	 * Must be called before the first resize()
	 */
	void use_g_buffers_of(FillGBufferRenderPass* g_buffers_owner);

private:
	/**
	 * Launches the kernel that returns the size of the RayVolumeState structure on the GPU for the compiler options of this
	 * render pass. Unlike get_ray_volume_state_byte_size(), the render data isn't updated
	 */
	size_t query_ray_volume_state_byte_size();
	/**
	 * Size of the elements of the RayVolumeState buffers of the G-buffers: the largest RayVolumeState of this render pass and of
	 * the render passes that fill its G-buffers. The render graphs have their own compiler options so the nested dielectrics stack,
	 * and with it the RayVolumeState, can have a different size in each of them
	 */
	size_t get_g_buffers_ray_volume_state_byte_size();

	/**
	 * Whether the G-buffer needs the lobe records of the MIS target function BSDF approximation of ReSTIR PT
	 * (GBufferDevice::mis_target_function_bsdf_approximation_lobe_records): ReSTIR PT path sampling with the union domain SPMIS reuse, the approximation
	 * enabled and the principled BSDF
	 */
	bool mis_target_function_bsdf_approximation_lobe_records_needed() const;

	int2_t m_render_resolution = make_int2(0, 0);

	// G-buffers of the current frame (camera rays hits) and previous frame
	GBufferGPURenderer m_g_buffer;
	GBufferGPURenderer m_g_buffer_prev_frame;

	// Render pass whose G-buffers this render pass fills, nullptr if this render pass fills its own G-buffers
	FillGBufferRenderPass* m_g_buffers_owner = nullptr;
	// Render passes that fill the G-buffers of this render pass
	std::vector<FillGBufferRenderPass*> m_g_buffers_users;

	// Kernel used for retrieving the size of the RayVolumeState structure on the GPU
	std::shared_ptr<GPUKernel> m_ray_volume_state_byte_size_kernel = nullptr;
	size_t m_ray_volume_state_byte_size							   = sizeof(RayVolumeState);

	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
};

#endif // #ifndef CAMERA_RAYS_RENDER_PASS_H
