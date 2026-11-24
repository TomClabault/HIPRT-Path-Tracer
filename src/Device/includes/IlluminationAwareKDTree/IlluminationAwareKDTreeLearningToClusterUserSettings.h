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
	float Q_prior_strength	  = 8.0f;

	int initial_sampling_budget_n0 = 4;
	// Direct multiplier to the probability of refinement of a node
	float refinement_aggressiveness		   = 4.0f;
	unsigned int refinement_stopping_gamma = 128;

	bool enable_lightcut_refinement = true;

	// The settings below only apply when the envmap is a cluster of the lightcuts, outside of ReSTIR PT.
	//
	// A shading point without a learned lightcut samples the envmap with its own NEE sample and the emissive lights with a second NEE sample
	// drawn from the lightcut without its envmap slot. The selection weight of the envmap in an unlearned lightcut is a prior that ignores
	// visibility: indoors, it would give most NEE samples to an occluded envmap
	bool envmap_separate_sampling_without_learned_lightcut = true;
	// At the primary hit of a pixel, chooses between the envmap and the emissive lights with a probability learned at that pixel from its
	// previous NEE samples, visibility included. A 3D spatial structure is too coarse for envmap visibility that changes over a few
	// centimeters, such as sunlight through a thin gap
	bool envmap_per_pixel_probability = true;
	// Smallest probability of either technique at a primary hit: a technique whose estimate is 0 by chance must keep being sampled, or the
	// light it can reach would be missed
	float envmap_per_pixel_probability_floor = 0.05f;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_USER_SETTINGS_H
