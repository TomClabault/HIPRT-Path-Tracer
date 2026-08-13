/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_KD_TREE_AND_NEE_DISTRIBUTIONS_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetBatchKDTreeAndNEEDistributionsStatistics(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int tree_cut_size, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ResetBatchKDTreeAndNEEDistributionsStatistics(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int tree_cut_size)
#endif
{
#ifdef __KERNELCC__
	unsigned int reset_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int reset_index = x;
#endif

	unsigned int node_count				 = *kd_tree_device.core.node_count;
	unsigned int distribution_slot_count = node_count * static_cast<unsigned int>(SurfaceNormalFace_Count) * tree_cut_size;

	if (reset_index == 0)
	{
		*kd_tree_device.core.training_sample_count					= 0;
		*kd_tree_device.nee_distributions.nee_training_record_count = 0;
	}

	if (reset_index < node_count)
	{
		kd_tree_device.core.batch_signatures[reset_index]	   = {};
		kd_tree_device.core.batch_spatial_moments[reset_index] = {};
	}

	if (reset_index < distribution_slot_count)
	{
		kd_tree_device.nee_distributions.batch_per_cut_node_second_moment_sum[reset_index] = 0.0f;
		kd_tree_device.nee_distributions.batch_per_cut_node_sample_count[reset_index]	   = 0;
	}
}

#endif
