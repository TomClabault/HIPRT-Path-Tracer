/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_H

#include "HostDeviceCommon/Maths/VecTypes.h"

struct IlluminationAwareKDTreeIlluminationSignatureDouble
{
	double valid_observation_count;
	double scalar_radiance_sum;
	double squared_scalar_radiance_sum;
};

struct IlluminationAwareKDTreeIlluminationSignature
{
	// Number of valid NEE observations, including zero-radiance samples.
	unsigned int valid_observation_count;

	// Sum of scalar incoming-radiance estimates.
	float scalar_radiance_sum;

	// Sum of squared scalar estimates.
	float squared_scalar_radiance_sum;

	// Radiance-weighted directional vector.
	float3_t weighted_direction_sum;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_SIGNATURE_H
