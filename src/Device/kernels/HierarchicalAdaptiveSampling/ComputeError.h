/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_COMPUTE_ERROR_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_COMPUTE_ERROR_H

#include "Device/includes/AdaptiveSampling.h"
#include "Device/kernels/HierarchicalAdaptiveSampling/Common.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) HierarchicalAdaptiveSamplingComputeError()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline HierarchicalAdaptiveSamplingComputeError(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA);
	unsigned int x				 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y				 = blockIdx.y * blockDim.y + threadIdx.y;
#endif // #ifdef __KERNELCC__
	int width  = render_data.render_settings.render_resolution.x;
	int height = render_data.render_settings.render_resolution.y;
	if (x >= width || y >= height)
		return;

	unsigned int pixel_index = x + y * width;
	int pixel_sample_count	 = render_data.aux_buffers.pixel_sample_count[pixel_index];
	float average_luminance;
	float confidence_interval = get_pixel_confidence_interval(render_data, pixel_index, pixel_sample_count, average_luminance);

	// Use the same absolute display-space uncertainty as per-pixel adaptive sampling so both thresholds share a numerical scale.
	render_data.aux_buffers.hierarchical_adaptive_sampling_error[pixel_index] =
		compute_adaptive_sampling_display_error(average_luminance, confidence_interval, render_data.display_post_process_settings);
}

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_COMPUTE_ERROR_H
