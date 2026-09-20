/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_DISPLAY_POST_PROCESS_H
#define KERNELS_DISPLAY_POST_PROCESS_H

#include "Device/includes/DisplayDebugViews.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Tonemapping.h"
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

	ColorRGB32F final_color;
	switch (display_settings.display_view)
	{
	case DISPLAY_POST_PROCESS_DENOISER_ALBEDO:
		final_color = display_view_denoiser_albedo(render_data, source_pixel_index);
		break;

	case DISPLAY_POST_PROCESS_DENOISER_NORMALS:
		final_color = display_view_denoiser_normals(render_data, source_pixel_index);
		if (display_settings.do_tonemapping == 1)
			final_color = tonemap_exponential(final_color, display_settings.exposure, display_settings.gamma);
		break;

	case DISPLAY_POST_PROCESS_DENOISED_BLEND:
		final_color = display_view_denoised_blend(render_data, source_pixel_index);
		break;

	case DISPLAY_POST_PROCESS_GMON_BLEND:
		final_color = display_view_gmon_blend(render_data, source_pixel_index);
		break;

	case DISPLAY_POST_PROCESS_WHITE_FURNACE_THRESHOLD:
	{
		bool has_adaptive_sampling_debug_color = display_view_compute_adaptive_sampling_debug_color(
			render_data, static_cast<int>(source_pixel_index), display_settings.adaptive_sampling_display_view,
			display_settings.adaptive_sampling_heatmap_index, final_color);
		if (!has_adaptive_sampling_debug_color)
		{
			ColorRGB32F* source_framebuffer =
				render_data.buffers.debug_ray_colors != nullptr ? render_data.buffers.debug_ray_colors : render_data.buffers.accumulated_ray_colors;
			unsigned int sample_count = static_cast<unsigned int>(hippt::max(1, display_settings.white_furnace_sample_count));
			final_color				  = source_framebuffer[source_pixel_index] / static_cast<float>(sample_count);
		}
		final_color = display_view_apply_white_furnace_threshold(final_color, display_settings);

		if (display_settings.do_tonemapping == 1)
			final_color = tonemap_exponential(final_color, display_settings.exposure, display_settings.gamma);
		break;
	}

	case DISPLAY_POST_PROCESS_DEFAULT:
	default:
	{
		bool has_adaptive_sampling_debug_color = display_view_compute_adaptive_sampling_debug_color(
			render_data, static_cast<int>(source_pixel_index), display_settings.adaptive_sampling_display_view,
			display_settings.adaptive_sampling_heatmap_index, final_color);
		if (!has_adaptive_sampling_debug_color)
		{
			ColorRGB32F* source_framebuffer =
				render_data.buffers.debug_ray_colors != nullptr ? render_data.buffers.debug_ray_colors : render_data.buffers.accumulated_ray_colors;
			unsigned int sample_count = hippt::max(1u, render_data.render_settings.sample_number);
			final_color				  = source_framebuffer[source_pixel_index] / static_cast<float>(sample_count);
			final_color.r			  = hippt::clamp(0.0f, 1.0e35f, final_color.r);
			final_color.g			  = hippt::clamp(0.0f, 1.0e35f, final_color.g);
			final_color.b			  = hippt::clamp(0.0f, 1.0e35f, final_color.b);
		}

		if (display_settings.do_tonemapping == 1)
			final_color = tonemap_exponential(final_color, display_settings.exposure, display_settings.gamma);
		break;
	}
	}

	render_data.buffers.display_post_processed_colors[output_pixel_index] = final_color;
}

#endif // #ifndef KERNELS_DISPLAY_POST_PROCESS_H
