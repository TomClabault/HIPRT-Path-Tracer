/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_COMMON_KERNELS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_COMMON_KERNELS_H

#include "Device/includes/Hash.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

HIPRT_DEVICE unsigned int compute_refinement_sampling_budget(const IlluminationAwareKDTreeLightClusteringData& cluster_data,
															 const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	float growth	 = static_cast<float>(cluster_data.cut_size) / static_cast<float>(settings.initial_light_cut_size);
	float multiplier = hippt::max(growth, 2.0f);

	return static_cast<unsigned int>(ceil(multiplier * static_cast<float>(settings.initial_sampling_budget_n0)));
}

HIPRT_DEVICE float compute_light_cluster_learning_rate(unsigned int iteration, const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	unsigned int time_step = iteration + 1u;

	return 1.0f / (settings.learning_rate_beta * hippt::intrin_pow(static_cast<float>(time_step), settings.learning_rate_omega));
}

HIPRT_DEVICE unsigned int get_light_cluster_iteration_budget(const IlluminationAwareKDTreeLightClusteringData& cluster_data,
															 const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	if (cluster_data.pending_record_budget > 0u)
		return hippt::min(cluster_data.pending_record_budget, static_cast<unsigned int>(IlluminationAwareKDTreePendingLightClusterRecordStride));

	return hippt::min(compute_refinement_sampling_budget(cluster_data, settings),
					  static_cast<unsigned int>(IlluminationAwareKDTreePendingLightClusterRecordStride));
}

HIPRT_DEVICE int find_light_cluster_slot(const IlluminationAwareKDTreeDevice& kd_tree, unsigned int clustering_index, unsigned int cluster_node_index)
{
	const IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];

	for (unsigned int slot = 0; slot < cluster_data.cut_size; slot++)
	{
		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		if (kd_tree.learning_to_cluster.light_cluster_node_indices[offset] == cluster_node_index)
			return static_cast<int>(slot);
	}

	return -1;
}

#endif
