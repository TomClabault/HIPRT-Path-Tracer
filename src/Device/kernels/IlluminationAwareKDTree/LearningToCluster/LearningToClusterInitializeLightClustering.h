/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTERING_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTERING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

HIPRT_DEVICE void learning_to_cluster_initialize_light_clustering_from_initial_cut(IlluminationAwareKDTreeDevice kd_tree,
																				   unsigned int lightcut_index,
																				   unsigned int slot)
{
	unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);

	if (slot < kd_tree.learning_to_cluster.effective_initial_lightcut_size)
		kd_tree.learning_to_cluster.lightcut_node_indices[offset] = kd_tree.learning_to_cluster.initial_lightcut_node_indices[slot];
	else
		kd_tree.learning_to_cluster.lightcut_node_indices[offset] = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	kd_tree.learning_to_cluster.lightcut_statistics[offset] = {};
	kd_tree.learning_to_cluster.lightcut_cdfs[offset]		= 0u;

	if (slot == 0)
	{
		IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
		lightcut_data											  = {};
		lightcut_data.lightcut_size								  = kd_tree.learning_to_cluster.effective_initial_lightcut_size;

		kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index]					 = 0;
		kd_tree.learning_to_cluster.lightcut_representative_shading_contexts[lightcut_index] = {};
		kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[lightcut_index] =
			IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTERING_H
