/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_USER_SETTINGS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_USER_SETTINGS_H

struct IlluminationAwareKDTreeLearningToClusterUserSettings
{
	unsigned int initial_light_cut_size = 4;

	// These parameters are kept internal to the implementation.
	float learning_rate_beta  = 4.0f;
	float learning_rate_omega = 6.0f / 7.0f;

	int initial_sampling_budget_n0		   = 16;
	unsigned int refinement_stopping_gamma = 128;

	bool enable_light_cut_refinement = true;
};

#endif
