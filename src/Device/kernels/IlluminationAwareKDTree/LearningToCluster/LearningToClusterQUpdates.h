/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/LearningToClusterCommon.h"

HIPRT_DEVICE void apply_replayed_aggregated_light_cluster_q_update(IlluminationAwareKDTreeLightClusterStatistics& statistics,
																   float learning_rate,
																   float history_weight,
																   float prior_strength,
																   float reward_sum,
																   unsigned int matching_record_count)
{
	if (matching_record_count == 0u)
		return;

	float batch_mean				= reward_sum / static_cast<float>(matching_record_count);
	statistics.learned_importance_Q = history_weight * statistics.learned_importance_Q + learning_rate * batch_mean;
	statistics.Q_observation_count += matching_record_count;

	float observation_count		   = static_cast<float>(statistics.Q_observation_count);
	float effective_prior_strength = hippt::max(0.0f, prior_strength);
	float normalization			   = observation_count + effective_prior_strength;

	statistics.estimated_importance_Q =
		(observation_count * statistics.learned_importance_Q + effective_prior_strength * statistics.prior_importance_Q) / normalization;
}

HIPRT_DEVICE void apply_replayed_light_cluster_batch_q_update(
	const IlluminationAwareKDTreeDevice& kd_tree, unsigned int lightcut_index, unsigned int slot, float learning_rate, float history_weight)
{
	unsigned int offset													= kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
	IlluminationAwareKDTreeLightClusterBatchStatistics batch_statistics = kd_tree.learning_to_cluster.lightcut_batch_statistics.read(offset);
	IlluminationAwareKDTreeLightClusterStatistics& statistics			= kd_tree.learning_to_cluster.lightcut_statistics[offset];

	apply_replayed_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, kd_tree.learning_to_cluster.user_settings.Q_prior_strength,
													 batch_statistics.contribution_sum, batch_statistics.selected_count);
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree)
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
	unsigned int replayed_sample_count						  = kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index];
	if (!lightcut_data.Q0_initialized)
		return;

	float learning_rate	 = compute_light_cluster_learning_rate(lightcut_data.iteration, kd_tree.learning_to_cluster.user_settings);
	float history_weight = 1.0f - learning_rate;

#ifdef __KERNELCC__
	if (slot < lightcut_data.lightcut_size)
		apply_replayed_light_cluster_batch_q_update(kd_tree, lightcut_index, slot, learning_rate, history_weight);

	__syncthreads();
	if (slot != 0u)
		return;
#else  // #ifdef __KERNELCC__
	for (unsigned int lightcut_slot = 0u; lightcut_slot < lightcut_data.lightcut_size; lightcut_slot++)
		apply_replayed_light_cluster_batch_q_update(kd_tree, lightcut_index, lightcut_slot, learning_rate, history_weight);
#endif // #ifdef __KERNELCC__

	if (replayed_sample_count > 0u)
	{
		lightcut_data.iteration++;
		lightcut_data.lightcut_cdf_dirty = true;
	}

	kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index] = 0u;
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_Q_UPDATES_H
