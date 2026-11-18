/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_SHADE_H
#define KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_SHADE_H

#include "Device/includes/ReSTIR/PT/Wavefront/WavefrontInitialCandidates.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_WAVEFRONT_SHADE_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) ReSTIR_PT_WavefrontInitialCandidatesShade(unsigned int bounce_count, unsigned int path_tree_index)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_PT_WavefrontInitialCandidatesShade(HIPRTRenderData render_data, unsigned int bounce_count, unsigned int path_tree_index, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_WAVEFRONT_SHADE_RENDER_DATA);

	unsigned int queue_slot	  = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int queue_stride = render_data.wavefront_data.path_capacity;
#endif // #ifdef __KERNELCC__

	// The trace kernel has appended the paths to the queue of the material family of their hit
	unsigned int material_family   = KERNEL_MATERIAL_SPECIALIZATION;
	unsigned int family_queue_base = material_family * render_data.wavefront_data.path_capacity;
	unsigned int queue_count	   = render_data.wavefront_data.material_family_counts[material_family];
	if (queue_count > render_data.wavefront_data.path_capacity)
		queue_count = render_data.wavefront_data.path_capacity;

	for (; queue_slot < queue_count; queue_slot += queue_stride)
	{
		unsigned int path_index = render_data.wavefront_data.material_family_indices[family_queue_base + queue_slot];
		if (path_index >= render_data.wavefront_data.path_capacity)
			continue;
		if (render_data.wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_TERMINAL)
			continue;

		ReSTIR_PT_wavefront_shade_path<false>(render_data, bounce_count, path_tree_index, path_index);
	}
}

#endif // #ifndef KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_SHADE_H
