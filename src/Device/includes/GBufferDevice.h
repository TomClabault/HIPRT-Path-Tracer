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
		return hippt::normalize(camera_position - primary_hit_position[pixel_index]);
	}

	DevicePackedEffectiveMaterial* materials = nullptr;

	int* first_hit_prim_index = nullptr;
	float3* primary_hit_position = nullptr;

	// We need both normals to correct the black fringes from the microfacet
	// model when used with smooth normals / normal mapping
	Octahedral24BitNormal* shading_normals = nullptr;
	Octahedral24BitNormal* geometric_normals = nullptr;
};

#endif