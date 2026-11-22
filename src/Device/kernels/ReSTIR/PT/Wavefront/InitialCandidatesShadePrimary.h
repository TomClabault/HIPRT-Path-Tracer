/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_SHADE_PRIMARY_H
#define KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_SHADE_PRIMARY_H

#include "Device/includes/ReSTIR/PT/Wavefront/WavefrontInitialCandidates.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_RENDER_DATA[sizeof(HIPRTRenderData)];
}

GLOBAL_KERNEL_SIGNATURE(void)
#if KERNEL_MATERIAL_SPECIALIZATION != 0
// Only keeping launch bounds for specialized shade primary and only removing it for the "All" material family Shade kernel
__launch_bounds__(64)
#endif // #if KERNEL_MATERIAL_SPECIALIZATION != 0
	ReSTIR_PT_WavefrontInitialCandidatesShadePrimary(unsigned int bounce_count, unsigned int path_tree_index)

#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_PT_WavefrontInitialCandidatesShadePrimary(HIPRTRenderData render_data,
														unsigned int bounce_count,
														unsigned int path_tree_index,
														unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_WAVEFRONT_SHADE_PRIMARY_RENDER_DATA);

	unsigned int queue_slot	  = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int queue_stride = render_data.wavefront_data.path_capacity;
#endif // #ifdef __KERNELCC__

	// The camera rays pass has appended the active pixels to the queue of the material family of their primary hit. The path trees after the
	// first one read a copy of these queues that the render pass restores before each of them
	unsigned int material_family   = KERNEL_MATERIAL_SPECIALIZATION;
	unsigned int family_queue_base = material_family * render_data.wavefront_data.path_capacity;

	unsigned int queue_count = render_data.wavefront_data.material_family_counts[material_family];
	if (queue_count > render_data.wavefront_data.path_capacity)
		queue_count = render_data.wavefront_data.path_capacity;

	for (; queue_slot < queue_count; queue_slot += queue_stride)
	{
		unsigned int pixel_index = render_data.wavefront_data.material_family_indices[family_queue_base + queue_slot];
		if (pixel_index >= render_data.wavefront_data.path_capacity)
			continue;

		ReSTIR_PT_wavefront_shade_path<true>(render_data, bounce_count, path_tree_index, pixel_index);
	}
}

#endif // #ifndef KERNELS_RESTIR_PT_WAVEFRONT_INITIAL_CANDIDATES_SHADE_PRIMARY_H
