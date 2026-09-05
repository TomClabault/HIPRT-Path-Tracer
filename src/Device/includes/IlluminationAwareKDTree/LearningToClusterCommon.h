/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_COMMON_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_COMMON_H

#include "Device/includes/Hash.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "HostDeviceCommon/IlluminationAwareKDTreeLearningToClusterStatisticsUpdateMode.h"

HIPRT_DEVICE unsigned int compute_refinement_sampling_budget(const IlluminationAwareKDTreeLightClusteringData& lightcut_data,
															 const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	float growth	 = static_cast<float>(lightcut_data.lightcut_size) / static_cast<float>(settings.initial_lightcut_size);
	float multiplier = hippt::max(growth, 2.0f);

	return static_cast<unsigned int>(ceil(multiplier * static_cast<float>(settings.initial_sampling_budget_n0)));
}

HIPRT_DEVICE float compute_light_cluster_learning_rate(unsigned int iteration, const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	unsigned int time_step = iteration + 1u;

	return 1.0f / (settings.learning_rate_beta * hippt::intrin_pow(static_cast<float>(time_step), settings.learning_rate_omega));
}

HIPRT_DEVICE int find_light_cluster_slot(const IlluminationAwareKDTreeDevice& kd_tree,
										 unsigned int lightcut_index,
										 unsigned int cluster_node_index,
										 unsigned int lightcut_slot)
{
	const IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];

	unsigned int base_offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0u);

	// Directly testing the slot of the sample itself first:
	if (lightcut_slot < lightcut_data.lightcut_size && kd_tree.learning_to_cluster.lightcut_node_indices[base_offset + lightcut_slot] == cluster_node_index)
		return static_cast<int>(lightcut_slot);

	for (unsigned int slot = 0; slot < lightcut_data.lightcut_size; slot++)
	{
		if (slot == lightcut_slot)
			// Already tested above
			continue;

		if (kd_tree.learning_to_cluster.lightcut_node_indices[base_offset + slot] == cluster_node_index)
			return static_cast<int>(slot);
	}

	return -1;
}

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_COMMON_H
