/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_RENDER_PASS_H
#define RENDERER_GMON_RENDER_PASS_H

#include "HostDeviceCommon/RenderData.h"
#include "Renderer/GPUDataStructures/GMoNGPUData.h"

class GMoNRenderPass
{
public:
	bool use_gmon();

	bool update(HIPRTRenderData& render_data);

	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> get_result_framebuffer();
	ColorRGB32F* get_sets_buffers_device_pointer();

	void resize_interop_buffers(unsigned int new_width, unsigned int new_height);
	void resize_non_interop_buffers(unsigned int new_width, unsigned int new_height);

	ColorRGB32F* map_result_framebuffer();
	void unmap_result_framebuffer();

private:
	// Data for the GMoN estimator
	GMoNGPUData m_gmon;
};

#endif
