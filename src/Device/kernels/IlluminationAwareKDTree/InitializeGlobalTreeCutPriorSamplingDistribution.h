/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_GLOBAL_TREE_CUT_PRIOR_SAMPLING_DISTRIBUTION_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_InitializeGlobalTreeCutPriorSamplingDistribution(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
																				LightTreeSGDevice light_tree_sg,
																				[[maybe_unused]] int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_InitializeGlobalTreeCutPriorSamplingDistribution(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
																		 LightTreeSGDevice light_tree_sg)
#endif
{
#ifndef __KERNELCC__
	unsigned int tree_cut_size				= light_tree_sg.settings.tree_cut_size;
	unsigned int guiding_distribution_index = illumination_aware_kd_tree.nodes[0].guiding_distribution_index;
	if (guiding_distribution_index >= illumination_aware_kd_tree.node_capacity)
		return;

	unsigned int tree_cut_offset = illumination_aware_kd_tree.nee_learnt_distributions.get_tree_cut_offset(guiding_distribution_index, tree_cut_size);

	unsigned int valid_node_count = 0;
	float total_power			  = 0.0f;
	for (unsigned int slot = 0; slot < tree_cut_size; slot++)
	{
		unsigned int node_index = light_tree_sg.tree_cut_node_indices[slot];
		if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			continue;

		valid_node_count++;
		total_power += hippt::max(light_tree_sg.nodes[node_index].total_power, 0.0f);
	}

	float uniform_probability	  = valid_node_count > 0 ? 1.0f / static_cast<float>(valid_node_count) : 0.0f;
	float running_cdf			  = 0.0f;
	unsigned int valid_slot_count = 0;
	for (unsigned int slot = 0; slot < tree_cut_size; slot++)
	{
		unsigned int node_index = light_tree_sg.tree_cut_node_indices[slot];
		float probability		= 0.0f;
		if (node_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		{
			float power				= hippt::max(light_tree_sg.nodes[node_index].total_power, 0.0f);
			float power_probability = total_power > 0.0f ? power / total_power : uniform_probability;
			probability				= power_probability;
			valid_slot_count++;
		}

		unsigned int distribution_slot = tree_cut_offset + slot;

		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot] = probability;
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot] =
			node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX ? 0.0f : running_cdf;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_estimated_second_moment[distribution_slot] = 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_sample_count[distribution_slot]			= 0;
		illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_second_moment_sum[distribution_slot]			= 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_sample_count[distribution_slot]				= 0;

		running_cdf += probability;
	}

	illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_sample_count[tree_cut_offset] = 0;
#else
	unsigned int slot = threadIdx.x;
	if (blockIdx.x != 0)
		return;

	unsigned int tree_cut_size				= light_tree_sg.settings.tree_cut_size;
	unsigned int invalid_node_index			= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int guiding_distribution_index = illumination_aware_kd_tree.nodes[0].guiding_distribution_index;
	unsigned int tree_cut_offset			= guiding_distribution_index * tree_cut_size;

	unsigned int tree_cut_node_index = slot < tree_cut_size ? light_tree_sg.tree_cut_node_indices[slot] : invalid_node_index;
	bool valid_slot					 = tree_cut_node_index != invalid_node_index;
	float power						 = 0.0f;
	if (valid_slot)
		power = hippt::max(light_tree_sg.nodes[tree_cut_node_index].total_power, 0.0f);

	unsigned int valid_node_count = block_reduce<IlluminationAwareKDTreeTreeCutInitializationBlockSize>(valid_slot ? 1u : 0u);
	float total_power			  = block_reduce<IlluminationAwareKDTreeTreeCutInitializationBlockSize>(power);
	float uniform_probability	  = valid_node_count > 0 ? 1.0f / static_cast<float>(valid_node_count) : 0.0f;
	float probability			  = 0.0f;
	if (valid_slot)
	{
		float power_probability = uniform_probability;
		if (total_power > 0.0f)
			power_probability = power / total_power;

		probability = power_probability;
	}

	float exclusive_cdf = block_prefix_scan_exclusive<IlluminationAwareKDTreeTreeCutInitializationBlockSize>(probability);
	if (slot < tree_cut_size)
	{
		unsigned int distribution_slot = tree_cut_offset + slot;

		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot]				= probability;
		illumination_aware_kd_tree.nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot]						= valid_slot ? exclusive_cdf : 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_estimated_second_moment[distribution_slot] = 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.history_per_cut_node_sample_count[distribution_slot]			= 0;
		illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_second_moment_sum[distribution_slot]			= 0.0f;
		illumination_aware_kd_tree.nee_learnt_distributions.batch_per_cut_node_sample_count[distribution_slot]				= 0;
	}

	illumination_aware_kd_tree.nee_learnt_distributions.history_per_cell_sample_count[tree_cut_offset] = 0;
#endif
}

#endif
