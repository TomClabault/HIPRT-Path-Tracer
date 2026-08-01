/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_TREE_CUT_SAMPLING_DISTRIBUTION_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_InitializeRootTreeCutSamplingDistribution(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
																		 unsigned int tree_cut_size,
																		 [[maybe_unused]] int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_InitializeRootTreeCutSamplingDistribution(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int tree_cut_size)
#endif
{
	if (tree_cut_size == 0)
		return;

	unsigned int guiding_distribution_index = illumination_aware_kd_tree.nodes[0].guiding_distribution_index;
	if (guiding_distribution_index >= illumination_aware_kd_tree.node_capacity)
		return;

	unsigned int tree_cut_offset = illumination_aware_kd_tree.nee_learnt_distributions.get_tree_cut_offset(guiding_distribution_index, tree_cut_size);

#ifndef __KERNELCC__
	for (unsigned int slot = 0; slot < tree_cut_size; slot++)
#else
	unsigned int slot = threadIdx.x;
	if (blockIdx.x != 0)
		return;

	if (slot < tree_cut_size)
#endif
	{
		float prior_probability		   = illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_prior_pdfs[slot];
		unsigned int distribution_slot = tree_cut_offset + slot;

		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_sample_count[0] = 0;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_estimated_second_moment[distribution_slot] =
			prior_probability * prior_probability;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_sample_count[distribution_slot] =
			IlluminationAwareKDTreeNEELearntDistributions::ROOT_PRIOR_STRENGTH;
		illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_second_moment_sum[distribution_slot] = 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_sample_count[distribution_slot]		= 0;
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot]		= prior_probability;
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot] =
			illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_prior_cdfs[slot];

		if (slot == 0)
			*illumination_aware_kd_tree.guiding_distribution_count = 1;
	}
}

#endif
