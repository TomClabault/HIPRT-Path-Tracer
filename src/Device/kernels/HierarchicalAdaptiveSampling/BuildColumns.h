/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_COLUMNS_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_COLUMNS_H

#include "Device/kernels/HierarchicalAdaptiveSampling/Common.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) HierarchicalAdaptiveSamplingBuildColumns()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline HierarchicalAdaptiveSamplingBuildColumns(HIPRTRenderData render_data, int x)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA);
	unsigned int x				 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	int width  = render_data.render_settings.render_resolution.x;
	int height = render_data.render_settings.render_resolution.y;
	if (x >= width)
		return;

	float column_sum = 0.0f;
	for (int y = 0; y < height; y++)
	{
		unsigned int pixel_index = x + y * width;
		column_sum += render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area[pixel_index];
		render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area[pixel_index] = column_sum;
	}
}

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_COLUMNS_H
