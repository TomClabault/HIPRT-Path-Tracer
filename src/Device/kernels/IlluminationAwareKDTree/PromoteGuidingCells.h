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
inline IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice kd_tree_device, unsigned long long int original_guiding_node_count, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice kd_tree_device, unsigned long long int original_guiding_node_count)
#endif
{
#ifdef __KERNELCC__
	unsigned int guiding_list_index = blockIdx.x;
#else
	unsigned int guiding_list_index = x;
#endif

	if (guiding_list_index >= original_guiding_node_count)
		return;

	if (kd_tree_device.core.needs_split[guiding_list_index] == 0)
		return;

	unsigned int parent_index = kd_tree_device.core.active_guiding_nodes[guiding_list_index];
	unsigned int node_count	  = *kd_tree_device.core.node_count;
	if (parent_index >= node_count)
		return;

	IlluminationAwareKDTreeNode& parent = kd_tree_device.core.nodes[parent_index];
	if (!(parent.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
		// We can only split parent who have children
		return;

	unsigned int left_child_index  = parent.left_child_index;
	unsigned int right_child_index = left_child_index + 1;

	IlluminationAwareKDTreeNode& left_child	 = kd_tree_device.core.nodes[left_child_index];
	IlluminationAwareKDTreeNode& right_child = kd_tree_device.core.nodes[right_child_index];

	__shared__ unsigned int active_guiding_output_index;
	__shared__ bool allocation_valid;
	// We launch one full 1024 threads block per each single cell. That's 1023 threads too many for 1 cell because we want a single atomic increment here so
	// only thread 0 does it and shares the result with the other threads of the block. And also only thread 0 does the memory writes
	if (threadIdx.x == 0)
	{
		// Replace the promoted guide with its left child and append the right child to the active guiding list so that's only 1 more allocated node
		active_guiding_output_index = hippt::atomic_fetch_add(kd_tree_device.core.active_guiding_node_count, 1u);
		allocation_valid			= active_guiding_output_index < kd_tree_device.core.node_capacity;
	}
	__syncthreads();

	if (!allocation_valid)
		return;

	if (threadIdx.x == 0)
	{
		left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
		right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;

		kd_tree_device.nisml.initialize_nisml_cache_for_guiding_cell(left_child_index, kd_tree_device.core.node_capacity);
		kd_tree_device.nisml.initialize_nisml_cache_for_guiding_cell(right_child_index, kd_tree_device.core.node_capacity);

		// The parent is no longer a guiding node
		parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;

		kd_tree_device.core.active_guiding_nodes[guiding_list_index]		  = left_child_index;
		kd_tree_device.core.active_guiding_nodes[active_guiding_output_index] = right_child_index;
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
			kd_tree_device.core.history_signatures[node_index] = kd_tree_device.core.batch_signatures[node_index];

			const IlluminationAwareKDTreeNode& node = kd_tree_device.core.nodes[node_index];
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
