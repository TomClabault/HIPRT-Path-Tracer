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
__launch_bounds__(64) WavefrontShadePaths(unsigned int bounce_count, bool route_by_family)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline WavefrontShadePaths(HIPRTRenderData render_data, unsigned int bounce_count, unsigned int queue_slot, bool route_by_family = false)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SHADE_RENDER_DATA);

	unsigned int queue_slot	  = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int queue_stride = render_data.wavefront_data.path_capacity;
#endif // #ifdef __KERNELCC__

	unsigned int queue_start = 0;
	unsigned int queue_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[0], 0u);
	if (route_by_family)
	{
		if (render_data.wavefront_data.material_family_routing_enabled == 0)
			return;

		unsigned int material_family = KERNEL_MATERIAL_SPECIALIZATION;
		queue_start					 = render_data.wavefront_data.material_family_offsets[material_family];
		queue_count					 = render_data.wavefront_data.material_family_counts[material_family];
	}
	else if (queue_count > render_data.wavefront_data.path_capacity)
		queue_count = render_data.wavefront_data.path_capacity;

	for (; queue_slot < queue_count; queue_slot += queue_stride)
	{
		unsigned int path_index = queue_slot;
		if (route_by_family)
			path_index = render_data.wavefront_data.material_family_indices[queue_start + queue_slot];
		else
			path_index = render_data.wavefront_data.path_queues[0][queue_slot];

		if (path_index >= render_data.wavefront_data.path_capacity)
			continue;
		if (render_data.wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_TERMINAL)
			continue;

		wavefront_shade_path<false>(render_data, bounce_count, path_index);
	}
}

#endif // #ifndef KERNELS_WAVEFRONT_SHADE_PATHS_H
