/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_TRAINING_SAMPLES_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_TRAINING_SAMPLES_H

#include "Device/includes/IlluminationAwareKDTree/LearningToClusterCommon.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

HIPRT_DEVICE unsigned int get_replayed_light_clustering_index(const IlluminationAwareKDTreeDevice& kd_tree,
															  const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample)
{
	if (sample.valid_for_lightcut == 0u)
		return IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

	return kd_tree.resolve_lightcut(sample.shading_context, sample.mesh_id);
}

HIPRT_DEVICE int find_replayed_light_cluster_slot_for_triangle(const IlluminationAwareKDTreeDevice& kd_tree,
															   const LightTreeSGDevice& light_tree_sg,
															   unsigned int lightcut_index,
															   int emissive_triangle_global_index)
{
	if (emissive_triangle_global_index < 0)
		return -1;

	unsigned int bit_trail			= light_tree_sg.bit_trails[emissive_triangle_global_index];
	unsigned int current_node_index = 0u;
	unsigned int current_depth		= 0u;

	while (true)
	{
		int slot = find_light_cluster_slot(kd_tree, lightcut_index, current_node_index);
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
												  unsigned int lightcut_index,
												  const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample)
{
	int slot = find_replayed_light_cluster_slot_for_triangle(kd_tree, light_tree_sg, lightcut_index, sample.emissive_triangle_global_index);
	if (slot >= 0)
		return slot;

	return find_light_cluster_slot(kd_tree, lightcut_index, sample.selected_cluster_node_index);
}

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_TRAINING_SAMPLES_H
