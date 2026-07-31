/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SAMPLED_CUT_NODE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SAMPLED_CUT_NODE_H

struct IlluminationAwareKDTreeSampledCutNode
{
	unsigned int cut_slot;
	unsigned int light_tree_node_index;
	float probability;
};

#endif
