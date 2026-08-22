/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTERING_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTERING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

HIPRT_DEVICE void initialize_light_clustering_from_initial_cut(IlluminationAwareKDTreeDevice kd_tree, unsigned int clustering_index, unsigned int slot)
{
	unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);

	if (slot < kd_tree.learning_to_cluster.effective_initial_light_cut_size)
		kd_tree.learning_to_cluster.light_cluster_node_indices[offset] = kd_tree.learning_to_cluster.initial_light_cut_node_indices[slot];
	else
		kd_tree.learning_to_cluster.light_cluster_node_indices[offset] = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	kd_tree.learning_to_cluster.light_cluster_statistics[offset] = {};
	kd_tree.learning_to_cluster.light_cluster_cdfs[offset]		 = 0u;
	kd_tree.learning_to_cluster.reservoir_proposals[offset]		 = 0ull;

	if (slot == 0)
	{
		IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
		cluster_data											 = {};
		cluster_data.cut_size									 = kd_tree.learning_to_cluster.effective_initial_light_cut_size;

		kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index] = 0;
		kd_tree.learning_to_cluster.reservoir_seen_counts[clustering_index]				  = 0;
		kd_tree.learning_to_cluster.representative_shading_contexts[clustering_index]	  = {};
		kd_tree.learning_to_cluster.representative_shading_context_states[clustering_index] =
			IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
	}
}

#endif
