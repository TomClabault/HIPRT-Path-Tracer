/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_RENDER_PASS_H
#define RENDERER_GMON_RENDER_PASS_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/GPUDataStructures/GMoNGPUData.h"

class GMoNRenderPass
{
public:
	GMoNRenderPass();
	GMoNRenderPass(GPURenderer* renderer);

	void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx);
	void recompile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, bool silent, bool use_cache);

	void launch();

	/**
	 * Allocates/deallocates the buffers used by GMoN.
	 * 
	 * Returns true or false depending on whether or not the render buffer data have been invalidated
	 */
	bool pre_render_update(HIPRTRenderData& render_data);
	void post_render_update(HIPRTRenderData& render_data);

	void reset();

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_result_framebuffer();
	ColorRGB32F* get_sets_buffers_device_pointer();
	unsigned int get_number_of_sets_used();

	void resize_interop_buffers(unsigned int new_width, unsigned int new_height);
	void resize_non_interop_buffers(unsigned int new_width, unsigned int new_height);

	ColorRGB32F* map_result_framebuffer();
	void unmap_result_framebuffer();

	bool use_gmon();

	GMoNGPUData& get_gmon_data();

private:
	GPURenderer* m_renderer = nullptr;

	// Data for the GMoN estimator
	GMoNGPUData m_gmon;

	GPUKernel m_compute_gmon_kernel;
};

#endif
