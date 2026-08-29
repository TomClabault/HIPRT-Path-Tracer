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
inline IlluminationAwareKDTree_CoreResetBatchKDTreeStatistics(IlluminationAwareKDTreeDevice kd_tree_device, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CoreResetBatchKDTreeStatistics(IlluminationAwareKDTreeDevice kd_tree_device)
#endif
{
#ifdef __KERNELCC__
	unsigned int reset_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int reset_index = x;
#endif

	unsigned int node_count = *kd_tree_device.core.node_count;
	if (reset_index == 0)
		*kd_tree_device.core.training_sample_count = 0;

	if (reset_index >= node_count)
		return;

	kd_tree_device.core.batch_signatures[reset_index]	   = {};
	kd_tree_device.core.batch_spatial_moments[reset_index] = {};
}

#endif
