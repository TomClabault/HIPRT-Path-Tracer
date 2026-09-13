/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_ROWS_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_ROWS_H

#include "Device/kernels/HierarchicalAdaptiveSampling/Common.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) HierarchicalAdaptiveSamplingBuildRows()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline HierarchicalAdaptiveSamplingBuildRows(HIPRTRenderData render_data, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA);
	unsigned int y				 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	int width  = render_data.render_settings.render_resolution.x;
	int height = render_data.render_settings.render_resolution.y;
	if (y >= height)
		return;

	float row_sum = 0.0f;
	for (int x = 0; x < width; x++)
	{
		unsigned int pixel_index = x + y * width;
		row_sum += render_data.aux_buffers.hierarchical_adaptive_sampling_error[pixel_index];
		render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area[pixel_index] = row_sum;
	}
}

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_ROWS_H
