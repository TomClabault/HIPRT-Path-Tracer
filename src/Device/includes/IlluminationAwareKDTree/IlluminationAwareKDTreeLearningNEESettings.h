/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_NEES_SETTINGS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_NEES_SETTINGS_H

struct IlluminationAwareKDTreeLearningNEESettings
{
	// A small amount of the global prior remains in every cell.
	//
	// This prevents a slot from permanently reaching zero probability which would be biased
	float minimum_global_prior_mix = 0.02f;

	// Maximum amount of the global tree cut that a cell can use
	float maximum_global_prior_mix = 0.50f;

	// Number of local observations required before the cell is considered well trained. This controls how quickly the global-prior mixture decreases.
	float local_evidence_scale = 256.0f;

	// Each child treats the parent's second-moment estimate as if it had received this many local observations per slot.
	unsigned int inherited_pseudo_count = 8;

	// Prevents the persistent count from becoming so large that the model can no longer adapt.
	float maximum_effective_count = 4096.0f;
};

#endif
