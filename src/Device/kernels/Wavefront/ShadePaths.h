/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_SHADE_PATHS_H
#define KERNELS_WAVEFRONT_SHADE_PATHS_H

#include "Device/includes/Wavefront/WavefrontShading.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_SHADE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) WavefrontShadePaths(unsigned int bounce_count)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline WavefrontShadePaths(HIPRTRenderData render_data, unsigned int bounce_count, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SHADE_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	unsigned int input_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[0], 0u);
	if (queue_slot >= input_count)
		return;

	unsigned int pixel_index = render_data.wavefront_data.path_queues[0][queue_slot];
	if (pixel_index >= render_data.wavefront_data.path_capacity)
		return;

	wavefront_shade_path<false>(render_data, bounce_count, pixel_index);
}

#endif // #ifndef KERNELS_WAVEFRONT_SHADE_PATHS_H
