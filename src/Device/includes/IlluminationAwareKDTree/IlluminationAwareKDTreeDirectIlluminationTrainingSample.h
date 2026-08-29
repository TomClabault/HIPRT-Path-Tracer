/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DIRECT_ILLUMINATION_TRAINING_SAMPLE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DIRECT_ILLUMINATION_TRAINING_SAMPLE_H

#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

#include <cstdint>

struct IlluminationAwareKDTreeDirectIlluminationTrainingSample
{
	float3_t position			= float3_t(0.0f, 0.0f, 0.0f);
	float3_t incoming_direction = float3_t(0.0f, 0.0f, 0.0f);

	float spatial_radiance_weight = 0.0f;

	unsigned int valid_for_spatial_training = false;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DIRECT_ILLUMINATION_TRAINING_SAMPLE_H
