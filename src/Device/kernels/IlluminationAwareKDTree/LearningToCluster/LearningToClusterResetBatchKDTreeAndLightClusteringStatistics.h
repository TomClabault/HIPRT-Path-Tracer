/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterResetBatchKDTreeAndLightClusteringStatistics(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, int x)
#else // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterResetBatchKDTreeAndLightClusteringStatistics(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int reset_index = blockIdx.x * blockDim.x + threadIdx.x;
#else // #ifdef __KERNELCC__
	unsigned int reset_index = x;
#endif // #ifdef __KERNELCC__

	unsigned int node_count = *illumination_aware_kd_tree.core.node_count;
	if (reset_index == 0)
	{
		*illumination_aware_kd_tree.core.training_sample_count				  = 0;
		*illumination_aware_kd_tree.learning_to_cluster.training_sample_count = 0;
	}

	if (reset_index >= node_count)
		return;

	if (reset_index < node_count)
	{
		illumination_aware_kd_tree.core.batch_signatures[reset_index]	   = {};
		illumination_aware_kd_tree.core.batch_spatial_moments[reset_index] = {};
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_H
