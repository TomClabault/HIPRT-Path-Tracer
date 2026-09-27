/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_H
#define KERNELS_WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_H

#include "Device/includes/Wavefront/WavefrontShading.h"

HIPRT_DEVICE static void wavefront_classify_secondary_material_family(HIPRTRenderData& render_data, unsigned int queue_slot, unsigned int input_count)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	if (queue_slot >= input_count)
		return;

	unsigned int path_index = wavefront_data.path_queues[0][queue_slot];
	if (path_index >= wavefront_data.path_capacity)
		return;
	if (wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_TERMINAL)
		return;

	KernelMaterialSpecialization material_family = KernelMaterialSpecializationAll;
	if (wavefront_data.path_state_flags[path_index] & WAVEFRONT_PATH_STATE_INTERSECTION_FOUND)
	{
		HitInfo& closest_hit_info = wavefront_data.path_closest_hit_infos[path_index];
		int material_index		  = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		ResolvedMaterialUserControlsCache resolved_user_controls;
		PrincipledMaterialClassificationInputs classification_inputs =
			load_material_classification_inputs(render_data, material_index, closest_hit_info.texcoords, resolved_user_controls);
		material_family = classify_principled_material(classification_inputs, !wavefront_data.path_volume_states[path_index].inside_material);

		// Keep the current hit separate from the previous vertex controls consumed by deferred MIS at the start of Shade.
		wavefront_store_current_material_classification(render_data, path_index, material_index, classification_inputs, resolved_user_controls);
	}

	unsigned int material_family_index = static_cast<unsigned int>(material_family);
	unsigned int family_queue_slot	   = hippt::atomic_fetch_add(wavefront_data.material_family_counts + material_family_index, 1u);
	if (family_queue_slot < wavefront_data.path_capacity)
		wavefront_data.material_family_indices[material_family_index * wavefront_data.path_capacity + family_queue_slot] = path_index;
}

#ifdef __KERNELCC__
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ClassifySecondaryMaterialFamilies()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ClassifySecondaryMaterialFamilies(HIPRTRenderData render_data, unsigned int queue_slot)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_RENDER_DATA);
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
		wavefront_classify_secondary_material_family(render_data, queue_slot, input_count);
}

#endif // #ifndef KERNELS_WAVEFRONT_CLASSIFY_SECONDARY_MATERIAL_FAMILIES_H
