/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_Q_REWARDS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_Q_REWARDS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/IlluminationAwareKDTree/ReplayLightClusterTrainingSamples.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterReplayQRewards(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else  // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterReplayQRewards(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else  // #ifdef __KERNELCC__
	unsigned int sample_index = static_cast<unsigned int>(x);
#endif // #ifdef __KERNELCC__

	unsigned int sample_count = *kd_tree.learning_to_cluster.training_sample_count;
	if (sample_index >= sample_count)
		return;

	unsigned int invalid_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
	unsigned int invalid_lightcut_slot	= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	unsigned int lightcut_index = kd_tree.learning_to_cluster.training_samples_soa.replayed_lightcut_indices[sample_index];
	unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
	if (lightcut_index == invalid_lightcut_index || lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;

	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	if (!lightcut_data.Q0_initialized)
		return;
	if (lightcut_data.batch_statistics_valid_for_q != 0u)
		return;

	const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];
	unsigned int lightcut_slot = kd_tree.learning_to_cluster.training_samples_soa.replayed_lightcut_slots[sample_index];
	unsigned int base_offset   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, 0u);

	bool cached_slot_matches = lightcut_slot < lightcut_data.lightcut_size &&
							   kd_tree.learning_to_cluster.lightcut_node_indices[base_offset + lightcut_slot] == sample.selected_cluster_node_index;
	if (!cached_slot_matches)
	{
		// The cached slot in replayed_lightcut_slots is invalid or does not match the selected cluster node index, so we need to resolve it again.
		int resolved_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, lightcut_index, sample);
		if (resolved_slot < 0)
			return;

		lightcut_slot = static_cast<unsigned int>(resolved_slot);
	}

	kd_tree.learning_to_cluster.training_samples_soa.replayed_lightcut_slots[sample_index] = static_cast<unsigned int>(lightcut_slot);

	unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, static_cast<unsigned int>(lightcut_slot));
	hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_batch_statistics.contribution_sum + offset, sample.q_reward);
	hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_batch_statistics.selected_count + offset, 1u);
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_Q_REWARDS_H
