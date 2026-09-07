/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_LIGHTCUT_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_LIGHTCUT_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterResetBatchLightcutStatistics(IlluminationAwareKDTreeDevice kd_tree, unsigned int reset_sample_counts, int x)
#else  // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterResetBatchLightcutStatistics(IlluminationAwareKDTreeDevice kd_tree, unsigned int reset_sample_counts)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int slot = threadIdx.x;
	for (unsigned int lightcut_index = blockIdx.x;; lightcut_index += gridDim.x)
	{
		unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
		if (lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
			break;

		if (slot >= LearningToClusterMaximumLightCutSize)
			continue;

		IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
		bool preserve_batch_statistics							  = reset_sample_counts == 0u && lightcut_data.batch_statistics_valid_for_q != 0u;
		if (preserve_batch_statistics)
			continue;

		if (reset_sample_counts != 0u && slot == 0u)
			kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index] = 0u;

		if (slot == 0u)
			lightcut_data.batch_statistics_valid_for_q = false;

		unsigned int lightcut_size = lightcut_data.lightcut_size;
		if (slot >= lightcut_size)
			continue;

		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		kd_tree.learning_to_cluster.lightcut_batch_statistics.reset(offset);
	}
#else  // #ifdef __KERNELCC__
	unsigned int lightcut_index = static_cast<unsigned int>(x);
	unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
	if (lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;

	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	if (reset_sample_counts == 0u && lightcut_data.batch_statistics_valid_for_q != 0u)
		return;

	if (reset_sample_counts != 0u)
		kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index] = 0u;
	lightcut_data.batch_statistics_valid_for_q = false;
#endif // #ifdef __KERNELCC__

#ifndef __KERNELCC__
	unsigned int lightcut_size = kd_tree.learning_to_cluster.lightcut_data[lightcut_index].lightcut_size;
	for (unsigned int lightcut_slot = 0u; lightcut_slot < lightcut_size; lightcut_slot++)
	{
		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, lightcut_slot);
		kd_tree.learning_to_cluster.lightcut_batch_statistics.reset(offset);
	}
#endif // #ifndef __KERNELCC__
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_LIGHTCUT_STATISTICS_H
