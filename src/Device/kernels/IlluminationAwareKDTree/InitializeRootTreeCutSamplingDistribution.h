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
inline IlluminationAwareKDTree_InitializeRootTreeCutSamplingDistribution(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int tree_cut_size, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_InitializeRootTreeCutSamplingDistribution(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int tree_cut_size)
#endif
{
#ifdef __KERNELCC__
	unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int slot = x;
#endif

	if (slot >= tree_cut_size)
		return;

	unsigned int guiding_distribution_index = kd_tree_device.core.nodes[0].guiding_distribution_index;
	if (guiding_distribution_index >= kd_tree_device.core.node_capacity)
		return;

	float prior_probability = static_cast<float>(kd_tree_device.nee_distributions.tree_cut_sampling_prior_pdfs[slot]) /
							  IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE;

	for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
	{
		unsigned int normal_face_distribution_index =
			kd_tree_device.nee_distributions.get_normal_face_distribution_index(guiding_distribution_index, normal_face);
		unsigned int tree_cut_offset   = kd_tree_device.nee_distributions.get_tree_cut_offset(normal_face_distribution_index, tree_cut_size);
		unsigned int distribution_slot = tree_cut_offset + slot;

		kd_tree_device.nee_distributions.history_per_cut_node_estimated_second_moment[distribution_slot] = prior_probability * prior_probability;
		kd_tree_device.nee_distributions.history_per_cut_node_sample_count[distribution_slot] =
			IlluminationAwareKDTreeNEELearntDistributionsDevice::ROOT_PRIOR_STRENGTH;
		kd_tree_device.nee_distributions.batch_per_cut_node_second_moment_sum[distribution_slot] = 0.0f;
		kd_tree_device.nee_distributions.batch_per_cut_node_sample_count[distribution_slot]		 = 0;
		kd_tree_device.nee_distributions.tree_cut_sampling_probabilities[distribution_slot] =
			kd_tree_device.nee_distributions.tree_cut_sampling_prior_pdfs[slot];
		kd_tree_device.nee_distributions.tree_cut_sampling_cdfs[distribution_slot] = kd_tree_device.nee_distributions.tree_cut_sampling_prior_cdfs[slot];

		if (slot == 0)
			kd_tree_device.nee_distributions.history_per_cell_sample_count[normal_face_distribution_index] = 0;
	}

	if (slot == 0)
		*kd_tree_device.core.guiding_distribution_count = 1;
}

#endif
