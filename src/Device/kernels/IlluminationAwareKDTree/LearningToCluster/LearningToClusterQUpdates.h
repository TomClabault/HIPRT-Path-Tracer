/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "Device/kernels/IlluminationAwareKDTree/ReplayLightClusterTrainingSamples.h"

HIPRT_DEVICE void apply_replayed_aggregated_light_cluster_q_update(IlluminationAwareKDTreeLightClusterStatistics& statistics,
																   float learning_rate,
																   float history_weight,
																   float reward_sum,
																   float reward_squared_sum,
																   unsigned int matching_record_count)
{
	if (matching_record_count == 0u)
		return;

#if LearningToClusterEstimateSecondMomentQ == KERNEL_OPTION_TRUE
	statistics.estimated_importance_Q = hippt::sqrt(history_weight * statistics.estimated_importance_Q * statistics.estimated_importance_Q +
													learning_rate * reward_squared_sum / static_cast<float>(matching_record_count));
#else
	statistics.estimated_importance_Q =
		history_weight * statistics.estimated_importance_Q + learning_rate * reward_sum / static_cast<float>(matching_record_count);
#endif
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
	unsigned int clustering_index		= blockIdx.x;
	unsigned int slot					= threadIdx.x;
	unsigned int light_clustering_count = *kd_tree.learning_to_cluster.light_clustering_count;
	if (clustering_index >= light_clustering_count || clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;
#else
	unsigned int clustering_index = static_cast<unsigned int>(x);
	unsigned int slot			  = 0u;
	if (clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;
#endif

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	unsigned int replayed_sample_count								= kd_tree.learning_to_cluster.light_cluster_sample_counts[clustering_index];
	if (!cluster_data.Q0_initialized)
		return;

	float learning_rate		  = compute_light_cluster_learning_rate(cluster_data.iteration, kd_tree.learning_to_cluster.user_settings);
	float history_weight	  = 1.0f - learning_rate;
	unsigned int sample_count = *kd_tree.learning_to_cluster.training_sample_count;

#ifdef __KERNELCC__
	if (slot < cluster_data.cut_size)
	{
		unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		float reward_sum										  = 0.0f;
		float reward_squared_sum								  = 0.0f;
		unsigned int matching_record_count						  = 0u;

		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];
			unsigned int sample_clustering_index								 = get_replayed_light_clustering_index(kd_tree, sample);
			if (sample_clustering_index != clustering_index)
				continue;

			int sample_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, clustering_index, sample);
			if (sample_slot != static_cast<int>(slot))
				continue;

			if (kd_tree.learning_to_cluster.user_settings.aggregate_q_updates)
			{
				reward_sum += sample.q_reward;
				reward_squared_sum += sample.q_reward * sample.q_reward;
				matching_record_count++;
			}
			else
			{
#if LearningToClusterEstimateSecondMomentQ == KERNEL_OPTION_TRUE
				statistics.estimated_importance_Q = hippt::sqrt(history_weight * statistics.estimated_importance_Q * statistics.estimated_importance_Q +
																learning_rate * sample.q_reward * sample.q_reward);
#else
				statistics.estimated_importance_Q = history_weight * statistics.estimated_importance_Q + learning_rate * sample.q_reward;
#endif
			}
		}

		if (kd_tree.learning_to_cluster.user_settings.aggregate_q_updates)
			apply_replayed_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, reward_sum, reward_squared_sum, matching_record_count);
	}

	__syncthreads();
	if (slot != 0u)
		return;
#else
	for (unsigned int cluster_slot = 0; cluster_slot < cluster_data.cut_size; cluster_slot++)
	{
		unsigned int offset											  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, cluster_slot);
		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		float reward_sum											  = 0.0f;
		float reward_squared_sum									  = 0.0f;
		unsigned int matching_record_count							  = 0u;

		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];
			unsigned int sample_clustering_index								 = get_replayed_light_clustering_index(kd_tree, sample);
			if (sample_clustering_index != clustering_index)
				continue;

			int sample_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, clustering_index, sample);
			if (sample_slot != static_cast<int>(cluster_slot))
				continue;

			if (kd_tree.learning_to_cluster.user_settings.aggregate_q_updates)
			{
				reward_sum += sample.q_reward;
				reward_squared_sum += sample.q_reward * sample.q_reward;
				matching_record_count++;
			}
			else
			{
#if LearningToClusterEstimateSecondMomentQ == KERNEL_OPTION_TRUE
				statistics.estimated_importance_Q = hippt::sqrt(history_weight * statistics.estimated_importance_Q * statistics.estimated_importance_Q +
																learning_rate * sample.q_reward * sample.q_reward);
#else
				statistics.estimated_importance_Q = history_weight * statistics.estimated_importance_Q + learning_rate * sample.q_reward;
#endif
			}
		}

		if (kd_tree.learning_to_cluster.user_settings.aggregate_q_updates)
			apply_replayed_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, reward_sum, reward_squared_sum, matching_record_count);
	}
#endif

	if (replayed_sample_count > 0u)
	{
		cluster_data.iteration++;
		cluster_data.light_cluster_cdf_dirty = true;
	}

	kd_tree.learning_to_cluster.light_cluster_sample_counts[clustering_index] = 0u;
}

#endif
