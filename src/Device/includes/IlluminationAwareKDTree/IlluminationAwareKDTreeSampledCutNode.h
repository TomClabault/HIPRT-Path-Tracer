/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SAMPLED_CUT_NODE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SAMPLED_CUT_NODE_H

struct IlluminationAwareKDTreeSampledCutNode
{
	// This value is used when sampling from a guiding distribution that has not been initialized yet. It is used to detect uninitialized distributions and
	// avoid using them for sampling.
	static constexpr float INVALID_PROBABILITY = -1.0f;

	unsigned int cut_slot;
	unsigned int light_tree_node_index;
	float probability = INVALID_PROBABILITY;
};

#endif
