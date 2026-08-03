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
												   unsigned long long int original_guiding_node_count,
												   int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned long long int original_guiding_node_count)
#endif
{
#ifdef __KERNELCC__
	unsigned int guiding_list_index = blockIdx.x; // *blockDim.x + threadIdx.x;
	unsigned int thread_slot		= threadIdx.x;
#else
	unsigned int guiding_list_index = x;
	unsigned int thread_slot		= 0;
#endif
	unsigned int slot_count = 1;

#ifndef __KERNELCC__
	slot_count = IlluminationAwareKDTreeMaximumLightCutSize;
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

	unsigned int parent_light_clustering_index = parent.light_clustering_index;

	IlluminationAwareKDTreeNode& left_child	 = illumination_aware_kd_tree.nodes[left_child_index];
	IlluminationAwareKDTreeNode& right_child = illumination_aware_kd_tree.nodes[right_child_index];

	__shared__ unsigned int right_light_clustering_index;
	__shared__ unsigned int active_guiding_output_index;
	__shared__ bool allocation_valid;
	// Only one thread allocates the right light clustering and active guiding-list entry. The result is shared with the rest of the block.
	if (thread_slot == 0)
	{
		// Only allocating 1 new light clustering for the right child, the left child will keep the parent's light clustering index
		right_light_clustering_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.light_clustering_count, 1u);

		// Replace the promoted guide with its left child and append the right child to the active guiding list so that's only 1 more allocated node
		active_guiding_output_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.active_guiding_node_count, 1u);

		allocation_valid =
			right_light_clustering_index < illumination_aware_kd_tree.node_capacity && active_guiding_output_index < illumination_aware_kd_tree.node_capacity;
	}
	__syncthreads();

	if (!allocation_valid)
		return;

	if (thread_slot == 0)
	{
		// The left child keeps the parent's light clustering index, the right child gets a new light clustering index but we will copy the parent's light
		// clustering into the right child so that it starts with the same light clustering as the left child (same as the parent)
		left_child.light_clustering_index  = parent_light_clustering_index;
		right_child.light_clustering_index = right_light_clustering_index;

		left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
		right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;

		// The parent is no longer a guiding node
		parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;
		parent.light_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

		illumination_aware_kd_tree.active_guiding_nodes[guiding_list_index]			 = left_child_index;
		illumination_aware_kd_tree.active_guiding_nodes[active_guiding_output_index] = right_child_index;
	}

	// We want thread 0 writes to be visible
	__syncthreads();

	for (unsigned int slot_iteration = 0; slot_iteration < slot_count; slot_iteration++)
	{
		unsigned int slot = thread_slot + slot_iteration;
		if (slot >= IlluminationAwareKDTreeMaximumLightCutSize)
			continue;

		unsigned int source_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(parent_light_clustering_index, slot);
		unsigned int right_offset  = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(right_light_clustering_index, slot);

		illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[right_offset] =
			illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[source_offset];
		illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[right_offset] =
			illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[source_offset];

		illumination_aware_kd_tree.learning_to_cluster.light_cluster_batch_statistics[source_offset] = {};
		illumination_aware_kd_tree.learning_to_cluster.light_cluster_batch_statistics[right_offset]	 = {};
	}

	if (thread_slot == 0)
	{
		illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_light_clustering_index] =
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_light_clustering_index];

		*(illumination_aware_kd_tree.learning_to_cluster.light_clustering_batch_sample_counts + parent_light_clustering_index) = 0;
		*(illumination_aware_kd_tree.learning_to_cluster.light_clustering_batch_sample_counts + right_light_clustering_index)  = 0;

		*(illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states + parent_light_clustering_index) = 0;
		*(illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states + right_light_clustering_index)	= 0;
	}

	__syncthreads();

	// The promoted subtree starts a fresh illumination-signature-history
	if (thread_slot == 0)
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
