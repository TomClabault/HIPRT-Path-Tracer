/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_PROMOTE_GUIDING_CELLS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_PROMOTE_GUIDING_CELLS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
												   unsigned int tree_cut_size,
												   unsigned long long int original_guiding_node_count,
												   int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
											unsigned int tree_cut_size,
											unsigned long long int original_guiding_node_count)
#endif
{
#ifdef __KERNELCC__
	unsigned int guiding_list_index = blockIdx.x; // *blockDim.x + threadIdx.x;
#else
	unsigned int guiding_list_index = x;
#endif

	if (guiding_list_index >= original_guiding_node_count)
		return;

	if (illumination_aware_kd_tree.needs_split[guiding_list_index] == 0)
		return;

	unsigned int parent_index = illumination_aware_kd_tree.active_guiding_nodes[guiding_list_index];
	unsigned int node_count	  = *illumination_aware_kd_tree.node_count;
	if (parent_index >= node_count)
		return;

	IlluminationAwareKDTreeNode& parent = illumination_aware_kd_tree.nodes[parent_index];
	if (!(parent.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
		// We can only split parent who have children
		return;

	unsigned int left_child_index  = parent.left_child_index;
	unsigned int right_child_index = left_child_index + 1;

	unsigned int parent_distribution_index = parent.guiding_distribution_index;

	IlluminationAwareKDTreeNode& left_child	 = illumination_aware_kd_tree.nodes[left_child_index];
	IlluminationAwareKDTreeNode& right_child = illumination_aware_kd_tree.nodes[right_child_index];

	__shared__ unsigned int right_distribution_index;
	__shared__ unsigned int active_guiding_output_index;
	__shared__ bool allocation_valid;
	// We launch one full 1024 threads block per each single cell. That's 1023 threads too many for 1 cell because we want a single atomic increment here so
	// only thread 0 does it and shares the result with the other threads of the block. And also only thread 0 does the memory writes
	if (threadIdx.x == 0)
	{
		// Only allocating 1 new distribution for the right child, the left child will keep the parent's distribution index
		right_distribution_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.guiding_distribution_count, 1u);

		// Replace the promoted guide with its left child and append the right child to the active guiding list so that's only 1 more allocated node
		active_guiding_output_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.active_guiding_node_count, 1u);

		allocation_valid =
			right_distribution_index < illumination_aware_kd_tree.node_capacity && active_guiding_output_index < illumination_aware_kd_tree.node_capacity;
	}
	__syncthreads();

	if (!allocation_valid)
		return;

	if (threadIdx.x == 0)
	{
		// The left child keeps the parent's distribution index, the right child gets a new distribution index but we will copy the parent's distribution into
		// the right child so that it starts with the same distribution as the left child (same as the parent)
		left_child.guiding_distribution_index  = parent_distribution_index;
		right_child.guiding_distribution_index = right_distribution_index;

		left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
		right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;

		// The parent is no longer a guiding node
		parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;
		parent.guiding_distribution_index = IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX;

		illumination_aware_kd_tree.active_guiding_nodes[guiding_list_index]			 = left_child_index;
		illumination_aware_kd_tree.active_guiding_nodes[active_guiding_output_index] = right_child_index;
	}

	// We want thread 0 writes to be visible
	__syncthreads();

	// Now we can finally use all the threads of the thread block correctly to copy the parent's distribution into the left and right child distribution
	IlluminationAwareKDTreeNEELearntDistributions& nee_learnt_distributions = illumination_aware_kd_tree.nee_learnt_distributions;

	unsigned int parent_distribution_offset		 = nee_learnt_distributions.get_tree_cut_offset(parent_distribution_index, tree_cut_size);
	unsigned int left_child_distribution_offset	 = parent_distribution_offset;
	unsigned int right_child_distribution_offset = nee_learnt_distributions.get_tree_cut_offset(right_child.guiding_distribution_index, tree_cut_size);

	if (threadIdx.x == 0)
	{
		// Fresh distributions have no history, so we reset the history sample counts
		nee_learnt_distributions.history_per_cell_sample_count[left_child.guiding_distribution_index]  = 0;
		nee_learnt_distributions.history_per_cell_sample_count[right_child.guiding_distribution_index] = 0;
	}

#ifndef __KERNELCC__
	unsigned int threads_per_block = 1;
#else
	unsigned int threads_per_block = blockDim.x;
#endif
	for (int slot_index = threadIdx.x; slot_index < tree_cut_size; slot_index += threads_per_block)
	{
		nee_learnt_distributions.history_per_cut_node_sample_count[right_child_distribution_offset + slot_index] =
			nee_learnt_distributions.learning_nee_settings.inherited_pseudo_count;
		nee_learnt_distributions.history_per_cut_node_sample_count[left_child_distribution_offset + slot_index] =
			nee_learnt_distributions.learning_nee_settings.inherited_pseudo_count;
		nee_learnt_distributions.history_per_cut_node_estimated_second_moment[right_child_distribution_offset + slot_index] =
			nee_learnt_distributions.history_per_cut_node_estimated_second_moment[parent_distribution_offset + slot_index];
		nee_learnt_distributions.history_per_cut_node_estimated_second_moment[left_child_distribution_offset + slot_index] =
			nee_learnt_distributions.history_per_cut_node_estimated_second_moment[parent_distribution_offset + slot_index];

		nee_learnt_distributions.tree_cut_sampling_probabilities[right_child_distribution_offset + slot_index] =
			nee_learnt_distributions.tree_cut_sampling_probabilities[parent_distribution_offset + slot_index];
		nee_learnt_distributions.tree_cut_sampling_probabilities[left_child_distribution_offset + slot_index] =
			nee_learnt_distributions.tree_cut_sampling_probabilities[parent_distribution_offset + slot_index];
		nee_learnt_distributions.tree_cut_sampling_cdfs[right_child_distribution_offset + slot_index] =
			nee_learnt_distributions.tree_cut_sampling_cdfs[parent_distribution_offset + slot_index];
		nee_learnt_distributions.tree_cut_sampling_cdfs[left_child_distribution_offset + slot_index] =
			nee_learnt_distributions.tree_cut_sampling_cdfs[parent_distribution_offset + slot_index];

		nee_learnt_distributions.batch_per_cut_node_sample_count[right_child_distribution_offset + slot_index]		= 0;
		nee_learnt_distributions.batch_per_cut_node_sample_count[left_child_distribution_offset + slot_index]		= 0;
		nee_learnt_distributions.batch_per_cut_node_second_moment_sum[right_child_distribution_offset + slot_index] = 0;
		nee_learnt_distributions.batch_per_cut_node_second_moment_sum[left_child_distribution_offset + slot_index]	= 0;
	}

	// The promoted subtree starts a fresh illumination-signature-history
	if (threadIdx.x == 0)
	{
		__shared__ unsigned int stack[128];

		unsigned int stack_size = 0;
		stack[stack_size++]		= left_child_index;
		stack[stack_size++]		= right_child_index;

		while (stack_size > 0)
		{
			unsigned int node_index = stack[--stack_size];

			// The new nodes start with the current batch signature as their history signature
			illumination_aware_kd_tree.history_signatures[node_index] = illumination_aware_kd_tree.batch_signatures[node_index];

			const IlluminationAwareKDTreeNode& node = illumination_aware_kd_tree.nodes[node_index];
			if (node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren)
			{
				unsigned int child_index = node.left_child_index;
				if (child_index + 1 < node_count && stack_size + 2 <= 128)
				{
					stack[stack_size++] = child_index;
					stack[stack_size++] = child_index + 1;
				}
			}
		}
	}
}

#endif
