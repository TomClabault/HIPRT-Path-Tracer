/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GBUFFER_DEVICE_H
#define GBUFFER_DEVICE_H

#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material/Material.h"

// Structure of arrays for the data contained in the pixels of the GBuffer
// 
// If you want the roughness of the pixel (X, Y) = [50, 0] for example,
// get it at materials[50].roughness
struct GBufferDevice
{
	HIPRT_HOST_DEVICE float3 get_view_direction(float3 camera_position, int pixel_index) const
	{
		return hippt::normalize(camera_position - first_hits[pixel_index]);
	}

	DevicePackedEffectiveMaterial* materials = nullptr;

	int* first_hit_prim_index = nullptr;

	// We need both normals to correct the blakc fringes from the microfacet
	// model when used with smooth normals / normal mapping
	float3* shading_normals = nullptr;
	float3* geometric_normals = nullptr;

	float3* first_hits = nullptr;

	unsigned char* camera_ray_hit = nullptr;

	RayVolumeState* ray_volume_states = nullptr;
};

#endif
