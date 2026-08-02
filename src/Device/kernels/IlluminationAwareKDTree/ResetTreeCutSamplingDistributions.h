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
inline IlluminationAwareKDTree_ResetTreeCutSamplingDistributions(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int tree_cut_size, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ResetTreeCutSamplingDistributions(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int tree_cut_size)
#endif
{
#ifdef __KERNELCC__
	unsigned int reset_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int reset_index = x;
#endif

	unsigned int distribution_slot_count = illumination_aware_kd_tree.node_capacity * tree_cut_size;
	if (reset_index >= distribution_slot_count)
		return;

	if (reset_index < illumination_aware_kd_tree.node_capacity)
	{
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_sample_count[reset_index] = 0;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_normal_sum_x[reset_index] = 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_normal_sum_y[reset_index] = 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_normal_sum_z[reset_index] = 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_normal_count[reset_index] = 0;
	}

	illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_probabilities[reset_index] =
		IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;
	illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_cdfs[reset_index] =
		IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_CDF_UNINITIALIZED_VALUE;
	illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_estimated_second_moment[reset_index] = 0.0f;
	illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_sample_count[reset_index]			  = 0;
	illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_second_moment_sum[reset_index]		  = 0.0f;
	illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_sample_count[reset_index]			  = 0;

	if (reset_index < tree_cut_size)
	{
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_prior_pdfs[reset_index] =
			IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_prior_cdfs[reset_index] =
			IlluminationAwareKDTreeNEELearntDistributions::TREE_CUT_SAMPLING_CDF_UNINITIALIZED_VALUE;
	}
}

#endif
