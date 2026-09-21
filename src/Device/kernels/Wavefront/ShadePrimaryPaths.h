/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_SHADE_PRIMARY_PATHS_H
#define KERNELS_WAVEFRONT_SHADE_PRIMARY_PATHS_H

#include "Device/includes/Wavefront/WavefrontShading.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_SHADE_PRIMARY_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) WavefrontShadePrimaryPaths(unsigned int bounce_count)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline WavefrontShadePrimaryPaths(HIPRTRenderData render_data, unsigned int bounce_count, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SHADE_PRIMARY_RENDER_DATA);
	unsigned int x				 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y				 = blockIdx.y * blockDim.y + threadIdx.y;
#endif // #ifdef __KERNELCC__
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	unsigned int pixel_index = x + y * render_data.render_settings.render_resolution.x;
	if (!render_data.aux_buffers.pixel_active[pixel_index])
		return;

	wavefront_shade_path<true>(render_data, bounce_count, pixel_index);
}

#endif // #ifndef KERNELS_WAVEFRONT_SHADE_PRIMARY_PATHS_H
