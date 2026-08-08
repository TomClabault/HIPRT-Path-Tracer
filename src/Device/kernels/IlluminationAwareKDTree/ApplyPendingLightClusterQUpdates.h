/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ApplyPendingLightClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ApplyPendingLightClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree)
#endif
{
#ifdef __KERNELCC__
	unsigned int active_pair_index = blockIdx.x;
	unsigned int slot			   = threadIdx.x;
#else
	unsigned int active_pair_index = static_cast<unsigned int>(x);
	unsigned int slot			   = 0;
#endif

	if (slot != 0u)
		return;

	unsigned int active_guiding_count = *kd_tree.active_guiding_node_count;
	if (active_pair_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_pair_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_pair_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	unsigned int clustering_index = kd_tree.learning_to_cluster.normal_clustering_sets[set_index].clustering_indices[normal_face];
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data			 = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;
	unsigned int pending_count											 = kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index];
	unsigned int iteration_budget = get_light_cluster_iteration_budget(cluster_data, settings);
	if (pending_count < iteration_budget)
		return;

	float learning_rate		 = compute_light_cluster_learning_rate(cluster_data.iteration, settings);
	float history_weight	 = 1.0f - learning_rate;
	unsigned int base_offset = clustering_index * kd_tree.learning_to_cluster.pending_record_stride;
	for (unsigned int record_index = 0; record_index < pending_count; record_index++)
	{
		const IlluminationAwareKDTreePendingLightClusterRecord& record = kd_tree.learning_to_cluster.pending_light_cluster_records[base_offset + record_index];
		int slot_index												   = find_light_cluster_slot(kd_tree, clustering_index, record.cluster_node_index);
		if (slot_index < 0)
			continue;

		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, static_cast<unsigned int>(slot_index));
		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		statistics.estimated_importance_Q						  = history_weight * statistics.estimated_importance_Q + learning_rate * record.q_reward;
	}

	cluster_data.iteration++;
	cluster_data.pending_record_budget = 0u;
	kd_tree.learning_to_cluster.reservoir_seen_counts[clustering_index] = 0u;
	kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index] = 0u;
}

#endif
