/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_RENDER_PASS_H
#define RENDERER_GMON_RENDER_PASS_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/GPUDataStructures/GMoNGPUData.h"
#include "Renderer/RenderPasses/RenderPass.h"
#include "UI/ApplicationSettings.h"

class GPURenderer;

class GMoNRenderPass : public RenderPass
{
public:
	static const std::string GMON_RENDER_PASS_NAME;
	static const std::string COMPUTE_GMON_KERNEL;

	GMoNRenderPass();
	GMoNRenderPass(GPURenderer* renderer);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	/**
	 * Allocates/deallocates the buffers used by GMoN.
	 * 
	 * Returns true or false depending on whether or not the render buffer data have been invalidated
	 */
	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch() override;

	float compute_gmon_darkening();
	float get_gmon_darkening();

	float get_lumi();

	/**
	 * Does the actual allocation/deallocation of the GMoN buffers.
	 * 
	 * Returns true a buffer was allocated or deallocated
	 * Returns false if buffer were left untouched
	 */
	virtual void post_render_update() override;

	virtual void update_render_data() override;
	virtual void reset() override;

	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

	virtual bool is_render_pass_used() const override;

	void request_recomputation();
	bool recomputation_completed();
	bool recomputation_requested();

	unsigned int get_last_recomputed_sample_count();

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_result_framebuffer();
	unsigned int get_number_of_sets_used();

	ColorRGB32F* map_result_framebuffer();
	void unmap_result_framebuffer();
	/**
	 * Returns true or false depending on whether or not the GMoN buffers are allocated
	 */
	bool buffers_allocated();

	GMoNGPUData& get_gmon_data();
	unsigned int get_VRAM_usage_bytes() const;

	float m_DEBUG_LUMINANCE_VARIANCE1 = 0.0f;
	float m_DEBUG_LUMINANCE_VARIANCE2 = 0.0f;
private:
	float m_darkening_factor = 0.0f;

	// Data for the GMoN estimator
	GMoNGPUData m_gmon;
};

#endif
