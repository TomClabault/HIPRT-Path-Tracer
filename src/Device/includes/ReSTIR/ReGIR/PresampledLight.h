/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_PRESAMPLED_LIGHT_H
#define DEVICE_KERNELS_REGIR_PRESAMPLED_LIGHT_H

#include "HostDeviceCommon/Packing.h"

struct ReGIRPresampledLight
{
	// Index in the whole scene of the triangle sampled 
	int emissive_triangle_index = -1;

	// Area of the sampled triangle
	float triangle_area = 0.0f;

	// Point sampled on the light
	float3 point_on_light = make_float3(0.0f, 0.0f, 0.0f);

	// Packed normal of the sampled emissive triangle
	Octahedral24BitNormalPadded32b normal;
};

#endif
