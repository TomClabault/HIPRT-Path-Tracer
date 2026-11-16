/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CUT_TRIANGLE_SAMPLE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CUT_TRIANGLE_SAMPLE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

struct IlluminationAwareKDTreeLearningToClusterCutTriangleSample
{
	int emissive_triangle_global_index = -1;

	// Guiding cell resolved while selecting the lightcut; reused by spatial training.
	unsigned int guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int lightcut_index		= IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
	unsigned int lightcut_slot		= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int cluster_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	unsigned int lightcut_size_at_sampling = 0;

	// p(c | x)
	float cluster_probability = 0.0f;

	// p(triangle | x, c)
	float conditional_triangle_probability = 0.0f;

	HIPRT_DEVICE float triangle_probability() const
	{
		return cluster_probability * conditional_triangle_probability;
	}

	/**
	 * Whether the selected cluster is the envmap. 'emissive_triangle_global_index' is then -1 and 'conditional_triangle_probability' is 1: the
	 * direction is sampled from the envmap distribution by the caller
	 */
	HIPRT_DEVICE bool is_envmap() const
	{
		return cluster_node_index == IlluminationAwareKDTreeNode::ENVMAP_CLUSTER_NODE_INDEX;
	}

	HIPRT_DEVICE bool valid() const
	{
		bool has_light = emissive_triangle_global_index >= 0 || is_envmap();

		return has_light && cluster_probability > 0.0f && conditional_triangle_probability > 0.0f;
	}
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CUT_TRIANGLE_SAMPLE_H
