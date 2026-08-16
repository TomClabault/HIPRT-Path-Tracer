/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CUT_TRIANGLE_SAMPLE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_CUT_TRIANGLE_SAMPLE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"
#include "Device/includes/FixIntellisense.h"

struct IlluminationAwareKDTreeLearningToClusterCutTriangleSample
{
	int emissive_triangle_global_index = -1;

	unsigned int light_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
	unsigned int cluster_slot			= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int cluster_node_index		= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	unsigned int cut_revision		  = 0;
	unsigned int cut_size_at_sampling = 0;

	// p(c | x)
	float cluster_probability = 0.0f;

	// p(triangle | x, c)
	float conditional_triangle_probability = 0.0f;

	HIPRT_DEVICE float triangle_probability() const
	{
		return cluster_probability * conditional_triangle_probability;
	}

	HIPRT_DEVICE bool valid() const
	{
		return emissive_triangle_global_index >= 0 && cluster_probability > 0.0f && conditional_triangle_probability > 0.0f;
	}
};

#endif
