/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_MATERIAL_FAMILY_ROUTING_H
#define DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_MATERIAL_FAMILY_ROUTING_H

#include "Device/includes/Material.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Maths/Math.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Appends 'path_index' to the queue of the 'material_family_index' material family.
 *
 * On the GPU, the lanes of a wave appending to the same family share a single atomic. All the family counters
 * live in the same cache line so one returning atomic per lane serializes heavily in the L2.
 *
 * The function can be called from divergent control flow: only the active lanes of the wave take part.
 */
HIPRT_DEVICE void wavefront_enqueue_material_family(HIPRTRenderData& render_data, unsigned int material_family_index, unsigned int path_index)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

	unsigned int family_queue_slot = 0;
#ifdef __KERNELCC__
	unsigned int lane_index					 = __lane_id();
	unsigned long long int lanes_before_mask = (1ull << lane_index) - 1ull;

	// Uniform loop over the families: each iteration reserves the slots of all the lanes of the wave that append to that family
	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		unsigned long long int family_lanes_mask = hippt::warp_ballot(0xFFFFFFFF, material_family_index == family_index);
		if (family_lanes_mask == 0)
			continue;

		unsigned int leader_lane = static_cast<unsigned int>(__ffsll(family_lanes_mask) - 1);

		unsigned int family_base_slot = 0;
		if (lane_index == leader_lane)
			family_base_slot = hippt::atomic_fetch_add(wavefront_data.material_family_counts + family_index, hippt::popc(family_lanes_mask));
		family_base_slot = hippt::warp_shfl(family_base_slot, leader_lane);

		if (material_family_index == family_index)
			family_queue_slot = family_base_slot + hippt::popc(family_lanes_mask & lanes_before_mask);
	}
#else  // #ifdef __KERNELCC__
	family_queue_slot = hippt::atomic_fetch_add(wavefront_data.material_family_counts + material_family_index, 1u);
#endif // #ifdef __KERNELCC__

	if (family_queue_slot < wavefront_data.path_capacity)
		wavefront_data.material_family_indices[material_family_index * wavefront_data.path_capacity + family_queue_slot] = path_index;
}

/**
 * Returns the material family of a primary hit.
 *
 * The classification uses the packed material written to the G-buffer so that the routing matches the material
 * that the primary shading pass reads back from the G-buffer.
 */
HIPRT_DEVICE unsigned int wavefront_classify_primary_material_family(const DevicePackedEffectiveMaterial& packed_material)
{
	ResolvedMaterialUserControlsCache resolved_user_controls;
	PrincipledMaterialClassificationInputs classification_inputs = load_material_classification_inputs(packed_material, resolved_user_controls);

	return static_cast<unsigned int>(classify_principled_material(classification_inputs, true));
}

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_MATERIAL_FAMILY_ROUTING_H
