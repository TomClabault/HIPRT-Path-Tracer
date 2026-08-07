/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

struct IlluminationAwareKDTreeLearningToClusterTrainingSample
{
	float3_t position = float3_t(0.0f, 0.0f, 0.0f);

	IlluminationAwareKDTreeSGShadingContext shading_context{};

	unsigned int selected_cluster_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	float cluster_probability  = 0.0f;
	float q_reward			   = 0.0f;
	float variance_observation = 0.0f;

	unsigned int sampled_light_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
	unsigned int selected_cluster_slot			= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int sampled_cut_revision			= 0;
	unsigned int sampled_cut_size				= 0;

	unsigned int valid_for_light_clustering = false;
};

#endif
