/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_APPLY_PENDING_LIGHT_CLUSTER_Q_UPDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

HIPRT_DEVICE void apply_aggregated_light_cluster_q_update(IlluminationAwareKDTreeLightClusterStatistics& statistics,
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
inline IlluminationAwareKDTree_ApplyPendingLightClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ApplyPendingLightClusterQUpdates(IlluminationAwareKDTreeDevice kd_tree)
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
	unsigned int slot			  = 0;
	if (clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;
#endif

	IlluminationAwareKDTreeLightClusteringData& cluster_data			 = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;
	unsigned int pending_count											 = kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index];
	unsigned int iteration_budget										 = get_light_cluster_iteration_budget(cluster_data, settings);
	if (pending_count < iteration_budget)
		// Waiting until we have enough pending records to update the light cluster statistics.
		return;

	float learning_rate		 = compute_light_cluster_learning_rate(cluster_data.iteration, settings);
	float history_weight	 = 1.0f - learning_rate;
	unsigned int base_offset = clustering_index * LearningToClusterMaximumClusterRecordCount;

#ifdef __KERNELCC__
	if (slot < cluster_data.cut_size)
	{
		unsigned int offset				= kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		unsigned int cluster_node_index = kd_tree.learning_to_cluster.light_cluster_node_indices[offset];

		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		float reward_sum										  = 0.0f;
		float reward_squared_sum								  = 0.0f;
		unsigned int matching_record_count						  = 0u;
		for (unsigned int record_index = 0; record_index < pending_count; record_index++)
		{
			const IlluminationAwareKDTreePendingLightClusterRecord& record =
				kd_tree.learning_to_cluster.pending_light_cluster_records[base_offset + record_index];

			if (record.cluster_node_index == cluster_node_index)
			{
				if (settings.aggregate_q_updates)
				{
					reward_sum += record.q_reward;
					reward_squared_sum += record.q_reward * record.q_reward;
					matching_record_count++;
				}
				else
				{
#if LearningToClusterEstimateSecondMomentQ == KERNEL_OPTION_TRUE
					statistics.estimated_importance_Q = hippt::sqrt(history_weight * statistics.estimated_importance_Q * statistics.estimated_importance_Q +
																	learning_rate * record.q_reward * record.q_reward);
#else
					statistics.estimated_importance_Q = history_weight * statistics.estimated_importance_Q + learning_rate * record.q_reward;
#endif
				}
			}
		}

		if (settings.aggregate_q_updates)
			apply_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, reward_sum, reward_squared_sum, matching_record_count);
	}

	__syncthreads();
	if (slot != 0u)
		return;
#else
	if (settings.aggregate_q_updates)
	{
		float reward_sums[LearningToClusterMaximumLightCutSize];
		float reward_squared_sums[LearningToClusterMaximumLightCutSize];
		unsigned int matching_record_counts[LearningToClusterMaximumLightCutSize];

		for (unsigned int slot = 0; slot < cluster_data.cut_size; slot++)
		{
			reward_sums[slot]			 = 0.0f;
			reward_squared_sums[slot]	 = 0.0f;
			matching_record_counts[slot] = 0u;
		}

		for (unsigned int record_index = 0; record_index < pending_count; record_index++)
		{
			const IlluminationAwareKDTreePendingLightClusterRecord& record =
				kd_tree.learning_to_cluster.pending_light_cluster_records[base_offset + record_index];
			int slot_index = find_light_cluster_slot(kd_tree, clustering_index, record.cluster_node_index);
			if (slot_index < 0)
				continue;

			unsigned int matching_slot_index = static_cast<unsigned int>(slot_index);

			reward_sums[matching_slot_index] += record.q_reward;
			reward_squared_sums[matching_slot_index] += record.q_reward * record.q_reward;
			matching_record_counts[matching_slot_index]++;
		}

		for (unsigned int slot = 0; slot < cluster_data.cut_size; slot++)
		{
			unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
			IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
			apply_aggregated_light_cluster_q_update(statistics, learning_rate, history_weight, reward_sums[slot], reward_squared_sums[slot],
													matching_record_counts[slot]);
		}
	}

	if (!settings.aggregate_q_updates)
	{
		for (unsigned int record_index = 0; record_index < pending_count; record_index++)
		{
			const IlluminationAwareKDTreePendingLightClusterRecord& record =
				kd_tree.learning_to_cluster.pending_light_cluster_records[base_offset + record_index];
			int slot_index = find_light_cluster_slot(kd_tree, clustering_index, record.cluster_node_index);
			if (slot_index < 0)
				continue;

			unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, static_cast<unsigned int>(slot_index));

			IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];

#if LearningToClusterEstimateSecondMomentQ == KERNEL_OPTION_TRUE
			statistics.estimated_importance_Q = hippt::sqrt(history_weight * statistics.estimated_importance_Q * statistics.estimated_importance_Q +
															learning_rate * record.q_reward * record.q_reward);
#else
			statistics.estimated_importance_Q = history_weight * statistics.estimated_importance_Q + learning_rate * record.q_reward;
#endif
		}
	}
#endif

	cluster_data.iteration++;
	cluster_data.light_cluster_cdf_dirty = true;
	cluster_data.pending_record_budget	 = 0u;

	kd_tree.learning_to_cluster.reservoir_seen_counts[clustering_index]				  = 0u;
	kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index] = 0u;
}

#endif
