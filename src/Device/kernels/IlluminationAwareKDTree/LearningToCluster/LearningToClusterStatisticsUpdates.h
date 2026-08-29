/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/LearningToClusterCommon.h"

HIPRT_DEVICE void merge_replayed_light_cluster_batch_statistics(const IlluminationAwareKDTreeDevice& kd_tree, unsigned int lightcut_index, unsigned int slot)
{
	unsigned int offset													= kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
	IlluminationAwareKDTreeLightClusterBatchStatistics batch_statistics = kd_tree.learning_to_cluster.lightcut_batch_statistics.read(offset);
	if (batch_statistics.selected_count == 0u)
		return;

	IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.lightcut_statistics[offset];
	float batch_mean										  = batch_statistics.contribution_sum / static_cast<float>(batch_statistics.selected_count);
	float batch_M2											  = batch_statistics.squared_contribution_sum - batch_statistics.contribution_sum * batch_mean;
	unsigned int previous_count								  = statistics.visit_count;
	if (previous_count == 0u)
	{
		statistics.mean		   = batch_mean;
		statistics.M2		   = batch_M2;
		statistics.visit_count = batch_statistics.selected_count;
		return;
	}

	unsigned int total_count = previous_count + batch_statistics.selected_count;
	float delta				 = batch_mean - statistics.mean;
	statistics.mean += delta * static_cast<float>(batch_statistics.selected_count) / static_cast<float>(total_count);
	statistics.M2 +=
		batch_M2 + delta * delta * static_cast<float>(previous_count) * static_cast<float>(batch_statistics.selected_count) / static_cast<float>(total_count);
	statistics.visit_count = total_count;
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterStatisticsUpdates(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterStatisticsUpdates(IlluminationAwareKDTreeDevice kd_tree)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int lightcut_index = blockIdx.x;
	unsigned int slot			= threadIdx.x;
	unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
	if (lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;
#else  // #ifdef __KERNELCC__
	unsigned int lightcut_index = static_cast<unsigned int>(x);
	if (lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;
#endif // #ifdef __KERNELCC__

	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	if (!lightcut_data.Q0_initialized)
		return;

#ifdef __KERNELCC__
	if (slot < lightcut_data.lightcut_size)
		merge_replayed_light_cluster_batch_statistics(kd_tree, lightcut_index, slot);
#else
	for (unsigned int lightcut_slot = 0u; lightcut_slot < lightcut_data.lightcut_size; lightcut_slot++)
		merge_replayed_light_cluster_batch_statistics(kd_tree, lightcut_index, lightcut_slot);
#endif // #ifdef __KERNELCC__
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_STATISTICS_H
