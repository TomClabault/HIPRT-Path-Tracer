/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_NEE_DISTRIBUTIONS_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_BATCH_NEE_DISTRIBUTIONS_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetBatchNEEDistributionsStatistics(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int tree_cut_size, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ResetBatchNEEDistributionsStatistics(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int tree_cut_size)
#endif
{
#ifdef __KERNELCC__
	unsigned int distribution_slot = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int distribution_slot = x;
#endif

	unsigned int node_count				 = *illumination_aware_kd_tree.node_count;
	unsigned int distribution_slot_count = node_count * tree_cut_size;
	if (distribution_slot >= distribution_slot_count)
		return;

	if (distribution_slot == 0)
		*illumination_aware_kd_tree.nee_learnt_distributions.nee_training_record_count = 0;

	illumination_aware_kd_tree.nee_learnt_distributions.batch_second_moment_sum[distribution_slot] = 0.0f;
	illumination_aware_kd_tree.nee_learnt_distributions.batch_sample_count[distribution_slot]	   = 0;
}

#endif
