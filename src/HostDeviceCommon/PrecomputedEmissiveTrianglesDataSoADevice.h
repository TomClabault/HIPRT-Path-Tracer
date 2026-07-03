/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_PRECOMPUTED_EMISSIVE_TRIANGLE_DATA_SOA_DEVICE_H
#define HOST_DEVICE_COMMON_PRECOMPUTED_EMISSIVE_TRIANGLE_DATA_SOA_DEVICE_H

struct PrecomputedEmissiveTrianglesDataSoADevice
{
	float3_t* triangles_A  = nullptr;
	float3_t* triangles_AB = nullptr;
	float3_t* triangles_AC = nullptr;
};

#endif
