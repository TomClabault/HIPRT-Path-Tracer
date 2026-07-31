/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetBatchKDTreeStatistics(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void) IlluminationAwareKDTree_ResetBatchKDTreeStatistics(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif
{
#ifdef __KERNELCC__
	unsigned int node_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int node_index = x;
#endif

	if (node_index >= illumination_aware_kd_tree.node_capacity)
		return;

	if (node_index == 0)
		*illumination_aware_kd_tree.training_sample_count = 0;

	illumination_aware_kd_tree.batch_signatures[node_index]		 = {};
	illumination_aware_kd_tree.batch_spatial_moments[node_index] = {};
}

#endif
