/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_SCATTER_SECONDARY_MATERIAL_FAMILIES_H
#define KERNELS_WAVEFRONT_SCATTER_SECONDARY_MATERIAL_FAMILIES_H

#include "Device/includes/Wavefront/WavefrontShading.h"

HIPRT_DEVICE static void wavefront_scatter_secondary_material_family(HIPRTRenderData& render_data, unsigned int queue_slot, unsigned int input_count)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	if (queue_slot >= input_count)
		return;

	unsigned int path_index = wavefront_data.path_queues[0][queue_slot];
	if (path_index >= wavefront_data.path_capacity)
		return;
	if (wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_TERMINAL)
		return;

	unsigned int material_family = wavefront_data.path_material_family_tags[path_index];
	if (material_family >= KernelMaterialSpecializationCount)
		return;

	unsigned int destination_index = hippt::atomic_fetch_add(wavefront_data.material_family_cursors + material_family, 1u);
	if (destination_index < wavefront_data.path_capacity)
		wavefront_data.material_family_indices[destination_index] = path_index;
}

#ifdef __KERNELCC__
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_SCATTER_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ScatterSecondaryMaterialFamilies()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ScatterSecondaryMaterialFamilies(HIPRTRenderData render_data, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SCATTER_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride	 = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int queue_stride = render_data.wavefront_data.path_capacity;
#endif // #ifdef __KERNELCC__
	if (render_data.wavefront_data.material_family_routing_enabled == 0)
		return;

	unsigned int input_count = hippt::atomic_fetch_add(render_data.wavefront_data.queue_counts[0], 0u);
	if (input_count > render_data.wavefront_data.path_capacity)
		input_count = render_data.wavefront_data.path_capacity;

	for (; queue_slot < input_count; queue_slot += queue_stride)
		wavefront_scatter_secondary_material_family(render_data, queue_slot, input_count);
}

#endif // #ifndef KERNELS_WAVEFRONT_SCATTER_SECONDARY_MATERIAL_FAMILIES_H
