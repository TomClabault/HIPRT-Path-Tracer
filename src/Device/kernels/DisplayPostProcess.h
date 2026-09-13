/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_DISPLAY_POST_PROCESS_H
#define KERNELS_DISPLAY_POST_PROCESS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/DisplayPostProcessSettings.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structures as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char DISPLAY_POST_PROCESS_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) DisplayPostProcess()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline DisplayPostProcess(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(DISPLAY_POST_PROCESS_RENDER_DATA);
	unsigned int x				 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y				 = blockIdx.y * blockDim.y + threadIdx.y;
#endif // #ifdef __KERNELCC__
	DisplayPostProcessSettings& display_settings = render_data.display_post_process_settings;

	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	unsigned int resolution_width	= render_data.render_settings.render_resolution.x;
	unsigned int resolution_scaling = render_data.render_settings.do_render_low_resolution() ? render_data.render_settings.render_low_resolution_scaling : 1;
	unsigned int source_x			= x / resolution_scaling;
	unsigned int source_y			= y / resolution_scaling;
	unsigned int source_pixel_index = source_x + source_y * resolution_width;
	unsigned int output_pixel_index = x + y * resolution_width;

	ColorRGB32F* source_framebuffer =
		render_data.buffers.debug_ray_colors != nullptr ? render_data.buffers.debug_ray_colors : render_data.buffers.accumulated_ray_colors;

	unsigned int sample_count = render_data.render_settings.sample_number + 1;
	ColorRGB32F final_color	  = source_framebuffer[source_pixel_index] / static_cast<float>(sample_count);
	final_color.r			  = hippt::clamp(0.0f, 1.0e35f, final_color.r);
	final_color.g			  = hippt::clamp(0.0f, 1.0e35f, final_color.g);
	final_color.b			  = hippt::clamp(0.0f, 1.0e35f, final_color.b);

	if (display_settings.do_tonemapping == 1)
	{
		ColorRGB32F tone_mapped = ColorRGB32F(1.0f) - intrin_expf(-final_color * display_settings.exposure);
		final_color				= intrin_pow(tone_mapped, 1.0f / display_settings.gamma);
	}

	render_data.buffers.display_post_processed_colors[output_pixel_index] = final_color;
}

#endif // #ifndef KERNELS_DISPLAY_POST_PROCESS_H
