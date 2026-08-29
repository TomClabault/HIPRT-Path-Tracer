/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_USER_SETTINGS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_USER_SETTINGS_H

#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeLearningToClusterOptions.h"

struct IlluminationAwareKDTreeLearningToClusterUserSettings
{
	unsigned int initial_lightcut_size = LearningToClusterInitialLightCutSize;

	float learning_rate_beta  = 1.0f;
	float learning_rate_omega = 6.0f / 7.0f;

	int initial_sampling_budget_n0		   = 32;
	unsigned int refinement_stopping_gamma = 128;

	bool enable_lightcut_refinement = true;
};

#endif
