/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_H

#include "HostDeviceCommon/Maths/VecTypes.h"

struct IlluminationAwareKDTreeSpatialSampleMoments
{
	// Only positive-radiance samples contribute here.
	unsigned int positive_radiance_sample_count = 0;

	float3_t position_sum		  = make_float3(0.0f, 0.0f, 0.0f);
	float3_t position_squared_sum = make_float3(0.0f, 0.0f, 0.0f);
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SPATIAL_SAMPLE_MOMENTS_H
