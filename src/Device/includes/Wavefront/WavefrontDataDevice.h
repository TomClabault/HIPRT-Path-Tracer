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
static constexpr unsigned int WAVEFRONT_COMPLETION_QUEUE_INDEX		  = 2u;

// Current-hit routing output. Previous-vertex controls remain in the resolved-control buffers until deferred MIS completes in Shade.
struct WavefrontResolvedMaterialClassification
{
	float roughness				= 0.0f;
	float metallic				= 0.0f;
	float specular				= 0.0f;
	float coat					= 0.0f;
	float sheen					= 0.0f;
	float specular_transmission = 0.0f;
	float dispersion_scale		= 0.0f;
};
static_assert(sizeof(WavefrontResolvedMaterialClassification) == sizeof(float) * 7,
			  "Wavefront material classification cache must remain a compact seven-float record");

struct WavefrontDataDevice
{
	ColorRGB32F* path_throughputs						= nullptr;
	ColorRGB32F* path_ray_colors						= nullptr;
	Octahedral24BitNormalPadded32b* path_ray_directions = nullptr;

	int* path_bounces					= nullptr;
	float* path_accumulated_roughnesses = nullptr;

	HitInfo* path_closest_hit_infos		 = nullptr;
	unsigned int* path_state_flags		 = nullptr;
	unsigned int* path_rng_states		 = nullptr;
	RayVolumeState* path_volume_states	 = nullptr;
	void* path_nee_deferred_mis_contexts = nullptr;
	// Each material family owns a fixed-capacity segment in this queue buffer.
	unsigned int* material_family_indices										   = nullptr;
	AtomicType<unsigned int>* material_family_counts							   = nullptr;
	float* path_resolved_material_roughness										   = nullptr;
	float* path_resolved_material_metallic										   = nullptr;
	float* path_resolved_material_specular										   = nullptr;
	float* path_resolved_material_coat											   = nullptr;
	float* path_resolved_material_sheen											   = nullptr;
	float* path_resolved_material_specular_transmission							   = nullptr;
	unsigned int* path_resolved_material_control_validity_masks					   = nullptr;
	WavefrontResolvedMaterialClassification* path_current_material_classifications = nullptr;

	unsigned int* path_queues[3]			  = { nullptr, nullptr, nullptr };
	AtomicType<unsigned int>* queue_counts[3] = { nullptr, nullptr, nullptr };

	unsigned int path_capacity					 = 0;
	unsigned int material_family_routing_enabled = 0;
};

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_DATA_DEVICE_H
