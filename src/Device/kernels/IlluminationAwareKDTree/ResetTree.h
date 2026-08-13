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
inline IlluminationAwareKDTree_ResetTree(IlluminationAwareKDTreeDevice kd_tree_device,
										 const float3_t scene_bounds_minimum,
										 const float3_t scene_bounds_maximum,
										 unsigned int node_index)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetTree(IlluminationAwareKDTreeDevice kd_tree_device, const float3 scene_bounds_minimum, const float3 scene_bounds_maximum)
#endif
{
#ifdef __KERNELCC__
	unsigned int node_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if (node_index >= kd_tree_device.core.node_capacity)
		return;

	if (node_index == 0)
	{
		IlluminationAwareKDTreeNode root{};
		root.flags = IlluminationAwareKDTreeNodeFlag_Guiding;

		kd_tree_device.core.nodes[0] = root;

		kd_tree_device.core.node_bounds[0].minimum = scene_bounds_minimum;
		kd_tree_device.core.node_bounds[0].maximum = scene_bounds_maximum;

		*kd_tree_device.core.node_count					= 1;
		*kd_tree_device.core.guiding_distribution_count = 1;
		*kd_tree_device.core.active_guiding_node_count	= 1;
		kd_tree_device.core.active_guiding_nodes[0]		= 0;

		*kd_tree_device.core.training_sample_count					= 0;
		*kd_tree_device.nee_distributions.nee_training_record_count = 0;
		if (kd_tree_device.nisml.nisml_pending_cell_count != nullptr)
			*kd_tree_device.nisml.nisml_pending_cell_count = ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT;

		*kd_tree_device.core.current_frontier_count = 0;
		*kd_tree_device.core.next_frontier_count	= 0;
	}

	kd_tree_device.core.history_signatures[node_index]		= {};
	kd_tree_device.core.history_spatial_moments[node_index] = {};

	kd_tree_device.core.batch_signatures[node_index]	  = {};
	kd_tree_device.core.batch_spatial_moments[node_index] = {};
	kd_tree_device.nisml.initialize_nisml_cache_for_guiding_cell(node_index, kd_tree_device.core.node_capacity);

	kd_tree_device.core.needs_split[node_index] = 0;
}

#endif
