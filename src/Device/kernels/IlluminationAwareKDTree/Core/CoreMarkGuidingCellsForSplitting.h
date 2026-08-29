/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_MARK_GUIDING_CELLS_FOR_SPLITTING_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_MARK_GUIDING_CELLS_FOR_SPLITTING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_CoreMarkGuidingCellsForSplitting(IlluminationAwareKDTreeDevice kd_tree_device, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CoreMarkGuidingCellsForSplitting(IlluminationAwareKDTreeDevice kd_tree_device)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int guiding_list_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int guiding_list_index = x;
#endif

	unsigned int active_guiding_count = *kd_tree_device.core.active_guiding_node_count;
	if (guiding_list_index >= active_guiding_count)
		return;

	unsigned int guiding_node_index = kd_tree_device.core.active_guiding_nodes[guiding_list_index];
	unsigned int node_count			= *kd_tree_device.core.node_count;
	if (guiding_node_index >= node_count)
	{
		kd_tree_device.core.needs_split[guiding_list_index] = 0;

		return;
	}

	IlluminationAwareKDTreeNode& guiding_node = kd_tree_device.core.nodes[guiding_node_index];
	if (!(guiding_node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren) ||
		guiding_node.left_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || guiding_node.left_child_index >= node_count ||
		guiding_node.left_child_index + 1u >= node_count)
	{
		kd_tree_device.core.needs_split[guiding_list_index] = 0;

		return;
	}

	unsigned int stack[128];
	unsigned int stack_size = 0;
	stack[stack_size++]		= guiding_node.left_child_index;
	stack[stack_size++]		= guiding_node.left_child_index + 1u;

	bool needs_split = false;
	while (stack_size > 0)
	{
		unsigned int lookahead_node_index = stack[--stack_size];
		if (lookahead_node_index >= node_count)
			continue;

		IlluminationAwareKDTreeSubdivisionMode subdivision_mode = kd_tree_device.core.user_settings.subdivision_mode;

		bool split_samples = false;
		if (subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::RECORD_SAMPLES_ONLY)
		{
			split_samples = kd_tree_device.core.should_split_samples(kd_tree_device.core.history_signatures[guiding_node_index]);
		}

		bool split_mean_radiance = false;
		if (subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::MEAN_RADIANCE_ONLY ||
			subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::FULL_MODEL)
		{
			split_mean_radiance = kd_tree_device.core.should_split_mean_radiance(kd_tree_device.core.history_signatures[guiding_node_index],
																				 kd_tree_device.core.history_signatures[lookahead_node_index]);
		}

		bool split_mean_direction = false;
		if (subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::MEAN_DIRECTION_ONLY ||
			subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::FULL_MODEL)
		{
			split_mean_direction = kd_tree_device.core.should_split_mean_direction(kd_tree_device.core.history_signatures[guiding_node_index],
																				   kd_tree_device.core.history_signatures[lookahead_node_index]);
		}

		if (split_samples || split_mean_radiance || split_mean_direction)
		{
			needs_split = true;

			break;
		}

		const IlluminationAwareKDTreeNode& lookahead_node = kd_tree_device.core.nodes[lookahead_node_index];
		if ((lookahead_node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren) && lookahead_node.left_child_index < node_count &&
			lookahead_node.left_child_index + 1u < node_count && stack_size + 2u <= 128u)
		{
			stack[stack_size++] = lookahead_node.left_child_index;
			stack[stack_size++] = lookahead_node.left_child_index + 1u;
		}
	}

	kd_tree_device.core.needs_split[guiding_list_index] = needs_split;
	if (needs_split)
		*kd_tree_device.any_cell_needs_split = 1;
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_MARK_GUIDING_CELLS_FOR_SPLITTING_H
