/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_DATA_DEVICE_H
#define DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_DATA_DEVICE_H

#include "Device/includes/HitInfo.h"
#include "Device/includes/RayVolumeState.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/Maths/VecTypes.h"
#include "HostDeviceCommon/Packing.h"

static constexpr unsigned int WAVEFRONT_PATH_STATE_INTERSECTION_FOUND = 1u;
static constexpr unsigned int WAVEFRONT_PATH_STATE_TERMINAL			  = 2u;

struct WavefrontDataDevice
{
	ColorRGB32F* path_throughputs						= nullptr;
	ColorRGB32F* path_ray_colors						= nullptr;
	Octahedral24BitNormalPadded32b* path_ray_directions = nullptr;

	int* path_bounces					= nullptr;
	float* path_accumulated_roughnesses = nullptr;

	HitInfo* path_closest_hit_infos								= nullptr;
	unsigned int* path_state_flags								= nullptr;
	unsigned int* path_rng_states								= nullptr;
	RayVolumeState* path_volume_states							= nullptr;
	void* path_nee_deferred_mis_contexts						= nullptr;
	unsigned int* path_material_family_tags						= nullptr;
	unsigned int* material_family_indices						= nullptr;
	AtomicType<unsigned int>* material_family_counts			= nullptr;
	unsigned int* material_family_offsets						= nullptr;
	AtomicType<unsigned int>* material_family_cursors			= nullptr;
	float* path_resolved_material_roughness						= nullptr;
	float* path_resolved_material_metallic						= nullptr;
	float* path_resolved_material_specular						= nullptr;
	float* path_resolved_material_coat							= nullptr;
	float* path_resolved_material_sheen							= nullptr;
	float* path_resolved_material_specular_transmission			= nullptr;
	unsigned int* path_resolved_material_control_validity_masks = nullptr;

	unsigned int* path_queues[2]			  = { nullptr, nullptr };
	AtomicType<unsigned int>* queue_counts[2] = { nullptr, nullptr };

	unsigned int path_capacity					 = 0;
	unsigned int material_family_routing_enabled = 0;
};

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_DATA_DEVICE_H
