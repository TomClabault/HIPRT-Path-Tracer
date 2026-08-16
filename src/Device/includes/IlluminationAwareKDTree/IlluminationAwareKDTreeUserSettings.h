/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_USER_SETTINGS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_USER_SETTINGS_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSubdivisionMode.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

struct IlluminationAwareKDTreeUserSettings
{
	int minimum_sample_count_for_lookahead_creation = 1000;
	int minimum_sample_count_for_splitting			= 1000;
	float mean_radiance_split_threshold				= 0.15f;

	int stop_refining_after_SPP = DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS ? 32 : 64;

	IlluminationAwareKDTreeSubdivisionMode subdivision_mode = IlluminationAwareKDTreeSubdivisionMode::FULL_MODEL;
};

#endif
