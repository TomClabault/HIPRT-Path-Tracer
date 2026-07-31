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
inline IlluminationAwareKDTreeDevice_MarkGuidingCellsForSplitting(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTreeDevice_MarkGuidingCellsForSplitting(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif
{
#ifdef __KERNELCC__
	const uint32_t guiding_list_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	const uint32_t guiding_list_index = x;
#endif

	const uint32_t active_guiding_count = *illumination_aware_kd_tree.active_guiding_node_count;
	if (guiding_list_index >= active_guiding_count)
		return;

	const uint32_t guiding_node_index = illumination_aware_kd_tree.active_guiding_nodes[guiding_list_index];
	const uint32_t node_count		  = *illumination_aware_kd_tree.node_count;
	if (guiding_node_index >= node_count)
	{
		illumination_aware_kd_tree.needs_split[guiding_list_index] = 0;

		return;
	}

	const IlluminationAwareKDTreeNode& guiding_node = illumination_aware_kd_tree.nodes[guiding_node_index];
	if (!(guiding_node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren) ||
		guiding_node.left_child_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || guiding_node.left_child_index >= node_count ||
		guiding_node.left_child_index + 1u >= node_count)
	{
		illumination_aware_kd_tree.needs_split[guiding_list_index] = 0;

		return;
	}

	uint32_t stack[128];
	uint32_t stack_size = 0;
	stack[stack_size++] = guiding_node.left_child_index;
	stack[stack_size++] = guiding_node.left_child_index + 1u;

	bool needs_split = false;
	while (stack_size > 0)
	{
		const uint32_t lookahead_node_index = stack[--stack_size];
		if (lookahead_node_index >= node_count)
			continue;

		IlluminationAwareKDTreeSubdivisionMode subdivision_mode = illumination_aware_kd_tree.user_settings.subdivision_mode;

		bool split_samples = false;
		if (subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::RECORD_SAMPLES_ONLY)
		{
			split_samples = illumination_aware_kd_tree.should_split_samples(illumination_aware_kd_tree.history_signatures[guiding_node_index]);
		}

		bool split_mean_radiance = false;
		if (subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::MEAN_RADIANCE_ONLY ||
			subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::FULL_MODEL)
		{
			split_mean_radiance = illumination_aware_kd_tree.should_split_mean_radiance(illumination_aware_kd_tree.history_signatures[guiding_node_index],
																						illumination_aware_kd_tree.history_signatures[lookahead_node_index]);
		}

		bool split_mean_direction = false;
		if (subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::MEAN_DIRECTION_ONLY ||
			subdivision_mode == IlluminationAwareKDTreeSubdivisionMode::FULL_MODEL)
		{
			split_mean_direction = illumination_aware_kd_tree.should_split_mean_direction(illumination_aware_kd_tree.history_signatures[guiding_node_index],
																						  illumination_aware_kd_tree.history_signatures[lookahead_node_index]);
		}

		if (split_samples || split_mean_radiance || split_mean_direction)
		{
			needs_split = true;

			break;
		}

		const IlluminationAwareKDTreeNode& lookahead_node = illumination_aware_kd_tree.nodes[lookahead_node_index];
		if ((lookahead_node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren) && lookahead_node.left_child_index < node_count &&
			lookahead_node.left_child_index + 1u < node_count && stack_size + 2u <= 128u)
		{
			stack[stack_size++] = lookahead_node.left_child_index;
			stack[stack_size++] = lookahead_node.left_child_index + 1u;
		}
	}

	illumination_aware_kd_tree.needs_split[guiding_list_index] = needs_split;
}

#endif
