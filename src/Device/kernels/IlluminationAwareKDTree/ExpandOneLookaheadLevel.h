/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_EXPAND_ONE_LOOKAHEAD_LEVEL_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_EXPAND_ONE_LOOKAHEAD_LEVEL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ExpandOneLookaheadLevel(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int creation_tag, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ExpandOneLookaheadLevel(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int creation_tag)
#endif
{
#ifdef __KERNELCC__
	unsigned int frontier_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int frontier_index = x;
#endif

	unsigned int current_frontier_count = *kd_tree_device.core.current_frontier_count;
	if (frontier_index >= current_frontier_count)
		return;

	unsigned int parent_index			= kd_tree_device.core.current_frontier[frontier_index];
	IlluminationAwareKDTreeNode& parent = kd_tree_device.core.nodes[parent_index];

	if (parent.flags & IlluminationAwareKDTreeNodeFlag_HasChildren)
	{
		// Existing physical children still form the next lookahead frontier.
		kd_tree_device.core.append_child_pair_to_frontier(kd_tree_device.core.next_frontier, kd_tree_device.core.next_frontier_count, parent.left_child_index,
														  parent.left_child_index + 1u);

		return;
	}

	// The paper waits until the parent has at least 1000 observations.
	if (kd_tree_device.core.history_signatures[parent_index].valid_observation_count <
		kd_tree_device.core.user_settings.minimum_sample_count_for_lookahead_creation)
		return;

	uint8_t split_axis	 = IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS;
	float split_position = 0.0f;
	kd_tree_device.core.compute_split_axis_and_position(kd_tree_device.core.history_spatial_moments[parent_index], split_axis, split_position);

	IlluminationAwareKDTreeNodeBounds& parent_bounds = kd_tree_device.core.node_bounds[parent_index];
	float minimum_split_extent = split_axis == 0 ? parent_bounds.minimum.x : (split_axis == 1 ? parent_bounds.minimum.y : parent_bounds.minimum.z);
	float maximum_split_extent = split_axis == 0 ? parent_bounds.maximum.x : (split_axis == 1 ? parent_bounds.maximum.y : parent_bounds.maximum.z);
	if (split_axis > 2 || !isfinite(split_position) || split_position <= minimum_split_extent || split_position >= maximum_split_extent)
		return;

	unsigned int left_child = kd_tree_device.core.reserve_physical_nodes(2u);
	if (left_child == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		// Not enough capacity to reserve
		return;

	unsigned int right_child = left_child + 1u;

	IlluminationAwareKDTreeNodeBounds left_bounds  = parent_bounds;
	IlluminationAwareKDTreeNodeBounds right_bounds = parent_bounds;
	if (split_axis == 0)
	{
		left_bounds.maximum.x  = split_position;
		right_bounds.minimum.x = split_position;
	}
	else if (split_axis == 1)
	{
		left_bounds.maximum.y  = split_position;
		right_bounds.minimum.y = split_position;
	}
	else
	{
		left_bounds.maximum.z  = split_position;
		right_bounds.minimum.z = split_position;
	}

	// Initialize the new child nodes with default values.
	IlluminationAwareKDTreeNode left_node{};
	left_node.left_child_index			   = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	left_node.creation_tag				   = creation_tag;
	left_node.split_axis				   = IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS;
	left_node.flags						   = IlluminationAwareKDTreeNodeFlag_Lookahead;
	IlluminationAwareKDTreeNode right_node = left_node;

	kd_tree_device.core.nodes[left_child]		 = left_node;
	kd_tree_device.core.nodes[right_child]		 = right_node;
	kd_tree_device.core.node_bounds[left_child]	 = left_bounds;
	kd_tree_device.core.node_bounds[right_child] = right_bounds;

	kd_tree_device.core.batch_signatures[left_child]		 = {};
	kd_tree_device.core.batch_signatures[right_child]		 = {};
	kd_tree_device.core.history_signatures[left_child]		 = {};
	kd_tree_device.core.history_signatures[right_child]		 = {};
	kd_tree_device.core.batch_spatial_moments[left_child]	 = {};
	kd_tree_device.core.batch_spatial_moments[right_child]	 = {};
	kd_tree_device.core.history_spatial_moments[left_child]	 = {};
	kd_tree_device.core.history_spatial_moments[right_child] = {};
	kd_tree_device.nisml.initialize_nisml_cache_for_guiding_cell(left_child, kd_tree_device.core.node_capacity);
	kd_tree_device.nisml.initialize_nisml_cache_for_guiding_cell(right_child, kd_tree_device.core.node_capacity);

	parent.split_axis		= split_axis;
	parent.split_position	= split_position;
	parent.left_child_index = left_child;
	parent.flags |= IlluminationAwareKDTreeNodeFlag_HasChildren;

	kd_tree_device.core.append_child_pair_to_frontier(kd_tree_device.core.next_frontier, kd_tree_device.core.next_frontier_count, left_child, right_child);
}

#endif
