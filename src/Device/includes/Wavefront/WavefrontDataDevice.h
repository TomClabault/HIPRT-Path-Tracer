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

struct WavefrontDataDevice
{
	ColorRGB32F* path_throughputs						= nullptr;
	ColorRGB32F* path_ray_colors						= nullptr;
	Octahedral24BitNormalPadded32b* path_ray_directions = nullptr;

	int* path_bounces					= nullptr;
	float* path_accumulated_roughnesses = nullptr;

	HitInfo* path_closest_hit_infos		   = nullptr;
	unsigned int* path_intersections_found = nullptr;
	unsigned int* path_rng_states		   = nullptr;
	RayVolumeState* path_volume_states	   = nullptr;
	void* path_nee_deferred_mis_contexts   = nullptr;

	unsigned int* path_queues[2]			  = { nullptr, nullptr };
	AtomicType<unsigned int>* queue_counts[2] = { nullptr, nullptr };

	unsigned int path_capacity = 0;
};

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_DATA_DEVICE_H
