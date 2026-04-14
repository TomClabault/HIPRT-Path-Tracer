/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PG_SPLATTING_SAMPLE_H
#define DEVICE_INCLUDES_RESTIR_PG_SPLATTING_SAMPLE_H

#include "Device/includes/HashGridHash.h"

#include "HostDeviceCommon/Maths/VecTypes.h"

struct ReSTIRPGSplattingSample
{
	// TODO can we just store the hash grid index directly instead of storing the information that we then rehash during the splatting pass?
	float3_t position;
	float3_t normal = make_float3(0.0f, 0.0f, 0.0f);
	float3_t incident_direction;

	HIPRT_DEVICE bool valid_sample() const
	{
		return normal.x != 0.0f || normal.y != 0.0f || normal.z != 0.0f;
	}
};

#endif
