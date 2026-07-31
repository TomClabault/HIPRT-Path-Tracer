/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_RESET_TREE_CUT_SAMPLING_DISTRIBUTIONS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetTreeCutSamplingDistributions(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
																 unsigned int tree_cut_size,
																 [[maybe_unused]] int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ResetTreeCutSamplingDistributions(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int tree_cut_size)
#endif
{
#ifndef __KERNELCC__
	unsigned int distribution_slot_count = illumination_aware_kd_tree.node_capacity * tree_cut_size;
	for (unsigned int distribution_slot = 0; distribution_slot < distribution_slot_count; distribution_slot++)
	{
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot] =
			IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot] =
			IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;
	}
#else
	unsigned int distribution_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int distribution_slot_count = illumination_aware_kd_tree.node_capacity * tree_cut_size;
	if (distribution_slot >= distribution_slot_count)
		return;

	illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot] =
		IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;
	illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot] =
		IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;
#endif
}

#endif
