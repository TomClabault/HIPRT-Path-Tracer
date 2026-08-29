/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/LearningToClusterCommon.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "Device/includes/IlluminationAwareKDTree/ReplayLightClusterTrainingSamples.h"

HIPRT_DEVICE void apply_replayed_aggregated_light_cluster_q_update(
	IlluminationAwareKDTreeLightClusterStatistics& statistics, float learning_rate, float history_weight, float reward_sum, unsigned int matching_record_count)
{
	if (matching_record_count == 0u)
		return;

	statistics.estimated_importance_Q =
		history_weight * statistics.estimated_importance_Q + learning_rate * reward_sum / static_cast<float>(matching_record_count);
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
#endif
{
#ifdef __KERNELCC__
	unsigned int lightcut_index = blockIdx.x;
	unsigned int slot			= threadIdx.x;
	unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
	if (lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;
#else
	unsigned int lightcut_index = static_cast<unsigned int>(x);
	unsigned int slot			= 0u;
	if (lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;
#endif

	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	unsigned int replayed_sample_count						  = kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index];
	if (!lightcut_data.Q0_initialized)
		return;

	float learning_rate		  = compute_light_cluster_learning_rate(lightcut_data.iteration, kd_tree.learning_to_cluster.user_settings);
	float history_weight	  = 1.0f - learning_rate;
	unsigned int sample_count = *kd_tree.learning_to_cluster.training_sample_count;

#ifdef __KERNELCC__
	if (slot < lightcut_data.lightcut_size)
	{
		unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.lightcut_statistics[offset];
		float reward_sum										  = 0.0f;
		unsigned int matching_record_count						  = 0u;

		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];
			unsigned int sample_lightcut_index									 = get_replayed_light_clustering_index(kd_tree, sample);
			if (sample_lightcut_index != lightcut_index)
				continue;

			int sample_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, lightcut_index, sample);
			if (sample_slot != static_cast<int>(slot))
				continue;

			reward_sum += sample.q_reward;
			matching_record_count++;
		}

		apply_replayed_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, reward_sum, matching_record_count);
	}

	__syncthreads();
	if (slot != 0u)
		return;
#else
	for (unsigned int lightcut_slot = 0; lightcut_slot < lightcut_data.lightcut_size; lightcut_slot++)
	{
		unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, lightcut_slot);
		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.lightcut_statistics[offset];
		float reward_sum										  = 0.0f;
		unsigned int matching_record_count						  = 0u;

		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];
			unsigned int sample_lightcut_index									 = get_replayed_light_clustering_index(kd_tree, sample);
			if (sample_lightcut_index != lightcut_index)
				continue;

			int sample_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, lightcut_index, sample);
			if (sample_slot != static_cast<int>(lightcut_slot))
				continue;

			reward_sum += sample.q_reward;
			matching_record_count++;
		}

		apply_replayed_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, reward_sum, matching_record_count);
	}
#endif

	if (replayed_sample_count > 0u)
	{
		lightcut_data.iteration++;
		lightcut_data.lightcut_cdf_dirty = true;
	}

	kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index] = 0u;
}

#endif
