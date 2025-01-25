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
#include "UI/ApplicationSettings.h"

class GMoNRenderPass
{
public:
	static const std::string COMPUTE_GMON_KERNEL;

	GMoNRenderPass();
	GMoNRenderPass(GPURenderer* renderer);

	void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx);
	void recompile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, bool silent, bool use_cache);

	void launch(std::shared_ptr<ApplicationSettings> application_settings);
	void request_recomputation();
	bool recomputation_completed();
	bool recomputation_requested();

	unsigned int get_last_recomputed_sample_count();

	/**
	 * Allocates/deallocates the buffers used by GMoN.
	 * 
	 * Returns true or false depending on whether or not the render buffer data have been invalidated
	 */
	bool pre_render_update();
	/**
	 * Does the actual allocation/deallocation of the GMoN buffers.
	 * 
	 * Returns true a buffer was allocated or deallocated
	 * Returns false if buffer were left untouched
	 */
	void post_render_update();

	void reset();

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_result_framebuffer();
	ColorRGB32F* get_sets_buffers_device_pointer();
	unsigned int get_number_of_sets_used();

	void resize_interop_buffers(unsigned int new_width, unsigned int new_height);
	void resize_non_interop_buffers(unsigned int new_width, unsigned int new_height);

	ColorRGB32F* map_result_framebuffer();
	void unmap_result_framebuffer();
	/**
	 * Returns true or false depending on whether or not the GMoN buffers are allocated
	 */
	bool buffers_allocated();

	bool use_gmon() const;

	GMoNGPUData& get_gmon_data();
	unsigned int get_VRAM_usage_bytes() const;

	std::map<std::string, std::shared_ptr<GPUKernel>> get_kernels();

private:
	GPURenderer* m_renderer = nullptr;

	// Data for the GMoN estimator
	GMoNGPUData m_gmon;

	std::map<std::string, std::shared_ptr<GPUKernel>> m_kernels;
};

#endif
