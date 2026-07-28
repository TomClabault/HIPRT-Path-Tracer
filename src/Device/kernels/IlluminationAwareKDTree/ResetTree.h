/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_NODE_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_NODE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "HostDeviceCommon/AtomicType.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetTree(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
										 const float3_t scene_bounds_minimum,
										 const float3_t scene_bounds_maximum)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetTree(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
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
	root.flags = IlluminationAwareKDTreeNodeFlag_Guiding;

	illumination_aware_kd_tree.nodes[0] = root;

	illumination_aware_kd_tree.node_bounds[0].minimum = scene_bounds_minimum;
	illumination_aware_kd_tree.node_bounds[0].maximum = scene_bounds_maximum;

	*illumination_aware_kd_tree.node_count				   = 1;
	*illumination_aware_kd_tree.active_guiding_node_count  = 1;
	illumination_aware_kd_tree.active_guiding_nodes[0]	   = 0;
	*illumination_aware_kd_tree.guiding_distribution_count = 1;
	*illumination_aware_kd_tree.training_sample_count	   = 0;

	*illumination_aware_kd_tree.current_frontier_count = 0;
	*illumination_aware_kd_tree.next_frontier_count	   = 0;

	illumination_aware_kd_tree.history_signatures[0]	  = {};
	illumination_aware_kd_tree.history_spatial_moments[0] = {};

	illumination_aware_kd_tree.batch_signatures[0]		= {};
	illumination_aware_kd_tree.batch_spatial_moments[0] = {};

	illumination_aware_kd_tree.needs_split[0] = 0;
}

#endif
