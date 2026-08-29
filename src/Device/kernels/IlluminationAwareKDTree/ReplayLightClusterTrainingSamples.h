/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_TRAINING_SAMPLES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_TRAINING_SAMPLES_H

#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

HIPRT_DEVICE unsigned int get_replayed_light_clustering_index(const IlluminationAwareKDTreeDevice& kd_tree,
															  const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample)
{
	if (sample.valid_for_light_clustering == 0u)
		return IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

	unsigned int guiding_node_index = kd_tree.core.find_guiding_cell(sample.position);
	if (guiding_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

	unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(sample.shading_context.shading_normal);
	unsigned int set_index	 = kd_tree.core.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

	return kd_tree.learning_to_cluster.normal_clustering_sets[set_index].clustering_indices[normal_face];
}

HIPRT_DEVICE int find_replayed_light_cluster_slot_for_triangle(const IlluminationAwareKDTreeDevice& kd_tree,
															   const LightTreeSGDevice& light_tree_sg,
															   unsigned int clustering_index,
															   int emissive_triangle_global_index)
{
	if (emissive_triangle_global_index < 0)
		return -1;

	unsigned int bit_trail			= light_tree_sg.bit_trails[emissive_triangle_global_index];
	unsigned int current_node_index = 0u;
	unsigned int current_depth		= 0u;

	while (true)
	{
		int slot = find_light_cluster_slot(kd_tree, clustering_index, current_node_index);
		if (slot >= 0)
			return slot;

		const LightTreeSGNodeDevice& current_node = light_tree_sg.nodes[current_node_index];
		if (current_node.triangle_count != 0u || current_depth >= sizeof(unsigned int) * 8u)
			return -1;

		unsigned int child_offset = (bit_trail & (1u << current_depth)) != 0u ? 1u : 0u;
		current_node_index		  = current_node.left_child_index_or_first_triangle_index + child_offset;
		current_depth++;
	}
}

HIPRT_DEVICE int find_replayed_light_cluster_slot(const IlluminationAwareKDTreeDevice& kd_tree,
												  const LightTreeSGDevice& light_tree_sg,
												  unsigned int clustering_index,
												  const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample)
{
	int slot = find_replayed_light_cluster_slot_for_triangle(kd_tree, light_tree_sg, clustering_index, sample.emissive_triangle_global_index);
	if (slot >= 0)
		return slot;

	return find_light_cluster_slot(kd_tree, clustering_index, sample.selected_cluster_node_index);
}

#endif
