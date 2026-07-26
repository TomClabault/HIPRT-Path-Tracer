/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_NODE_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_NODE_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/IlluminationAwareKDTreeNodeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline initialize_illumination_tree_root(IlluminationAwareKDTreeNode* nodes,
										 IlluminationAwareKDTreeNodeBounds* bounds,
										 uint32_t* node_count,
										 uint32_t* active_guiding_nodes,
										 uint32_t* active_guiding_node_count,
										 const float3_t scene_bounds_minimum,
										 const float3_t scene_bounds_maximum)
#else
GLOBAL_KERNEL_SIGNATURE(void)
initialize_illumination_tree_root(IlluminationAwareKDTreeNode* nodes,
								  IlluminationAwareKDTreeNodeBounds* bounds,
								  uint32_t* node_count,
								  uint32_t* active_guiding_nodes,
								  uint32_t* active_guiding_node_count,
								  const float3 scene_bounds_minimum,
								  const float3 scene_bounds_maximum)
#endif
{
#ifdef __KERNELCC__
	// Only one thread initializes the root.
	if (blockIdx.x != 0 || threadIdx.x != 0)
		return;
#endif

	IlluminationAwareKDTreeNode root{};

	// The root starts without lookahead children.
	root.left_child_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	// The root is initially the only active guiding cell.
	root.guiding_distribution_index = 0;

	// The root was not created by a replayable lookahead pass.
	//
	// Keeping this invalid prevents the root from accidentally receiving
	// the same current-batch sample again during a replay pass.
	root.creation_tag = IlluminationAwareKDTreeNode::INVALID_CREATION_TAG;

	root.split_position = 0.0f;
	root.split_axis		= 0;
	root.flags			= IlluminationAwareKDTreeNodeFlag_Guiding;

	nodes[0] = root;

	bounds[0].minimum = scene_bounds_minimum;
	bounds[0].maximum = scene_bounds_maximum;

	*node_count				   = 1;
	active_guiding_nodes[0]	   = 0;
	*active_guiding_node_count = 1;
}

#endif
