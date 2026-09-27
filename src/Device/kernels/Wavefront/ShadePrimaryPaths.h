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
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) WavefrontShadePrimaryPaths(unsigned int bounce_count, bool route_by_family)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline WavefrontShadePrimaryPaths(HIPRTRenderData render_data, unsigned int bounce_count, bool route_by_family, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SHADE_PRIMARY_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride	 = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int queue_slot	  = static_cast<unsigned int>(x + y * render_data.render_settings.render_resolution.x);
	unsigned int queue_stride = 1;
#endif // #ifdef __KERNELCC__

	unsigned int queue_count = render_data.wavefront_data.path_capacity;
	if (route_by_family)
	{
		if (render_data.wavefront_data.material_family_routing_enabled == 0)
			return;

		unsigned int material_family = KERNEL_MATERIAL_SPECIALIZATION;
		queue_count					 = render_data.wavefront_data.material_family_counts[material_family];
	}
	if (queue_count > render_data.wavefront_data.path_capacity)
		queue_count = render_data.wavefront_data.path_capacity;

	for (; queue_slot < queue_count; queue_slot += queue_stride)
	{
		unsigned int pixel_index = queue_slot;
		if (route_by_family)
		{
			unsigned int material_family   = KERNEL_MATERIAL_SPECIALIZATION;
			unsigned int family_queue_base = material_family * render_data.wavefront_data.path_capacity;
			pixel_index					   = render_data.wavefront_data.material_family_indices[family_queue_base + queue_slot];
		}
		else if (!render_data.aux_buffers.pixel_active[pixel_index])
			continue;

		if (pixel_index >= render_data.wavefront_data.path_capacity)
			continue;

		wavefront_shade_path<true>(render_data, bounce_count, pixel_index);
	}
}

#endif // #ifndef KERNELS_WAVEFRONT_SHADE_PRIMARY_PATHS_H
