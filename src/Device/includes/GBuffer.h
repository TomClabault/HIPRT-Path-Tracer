/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GBUFFER_H
#define DEVICE_GBUFFER_H

#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material.h"

// Structure of arrays for the data contained in the pixels of the GBuffer
// 
// If you want the roughness of the pixel (X, Y) = [50, 0] for example,
// get it at materials[50].roughness
struct GBuffer
{
	DeviceEffectiveMaterial* materials = nullptr;

	int* first_hit_prim_index = nullptr;

	// We need both normals to correct the blakc fringes from the microfacet
	// model when used with smooth normals / normal mapping
	float3* shading_normals = nullptr;
	float3* geometric_normals = nullptr;

	float3* view_directions = nullptr;
	float3* first_hits = nullptr;

	unsigned char* camera_ray_hit = nullptr;

	RayVolumeState* ray_volume_states = nullptr;
};

#endif
