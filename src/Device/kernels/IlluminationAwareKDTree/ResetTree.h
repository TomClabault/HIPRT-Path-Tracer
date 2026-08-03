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
										 const float3_t scene_bounds_maximum,
										 unsigned int node_index)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetTree(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
										 const float3 scene_bounds_minimum,
										 const float3 scene_bounds_maximum)
#endif
{
#ifdef __KERNELCC__
	unsigned int node_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if (node_index >= illumination_aware_kd_tree.node_capacity)
		return;

	if (node_index == 0)
	{
		IlluminationAwareKDTreeNode root{};
		root.flags					= IlluminationAwareKDTreeNodeFlag_Guiding;
		root.light_clustering_index = 0;

		illumination_aware_kd_tree.nodes[0] = root;

		illumination_aware_kd_tree.node_bounds[0].minimum = scene_bounds_minimum;
		illumination_aware_kd_tree.node_bounds[0].maximum = scene_bounds_maximum;

		*illumination_aware_kd_tree.node_count								   = 1;
		*illumination_aware_kd_tree.learning_to_cluster.light_clustering_count = 1;
		*illumination_aware_kd_tree.active_guiding_node_count				   = 1;
		illumination_aware_kd_tree.active_guiding_nodes[0]					   = 0;

		*illumination_aware_kd_tree.training_sample_count					  = 0;
		*illumination_aware_kd_tree.learning_to_cluster_training_sample_count = 0;

		*illumination_aware_kd_tree.current_frontier_count = 0;
		*illumination_aware_kd_tree.next_frontier_count	   = 0;
	}

	illumination_aware_kd_tree.history_signatures[node_index]	   = {};
	illumination_aware_kd_tree.history_spatial_moments[node_index] = {};

	illumination_aware_kd_tree.batch_signatures[node_index]		 = {};
	illumination_aware_kd_tree.batch_spatial_moments[node_index] = {};

	illumination_aware_kd_tree.needs_split[node_index] = 0;

	illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[node_index]				= {};
	illumination_aware_kd_tree.learning_to_cluster.light_clustering_batch_sample_counts[node_index] = 0;
	illumination_aware_kd_tree.learning_to_cluster.representative_shading_contexts[node_index]		= {};
	illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states[node_index] =
		IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

	for (unsigned int slot = 0; slot < IlluminationAwareKDTreeMaximumLightCutSize; slot++)
	{
		unsigned int cluster_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(node_index, slot);
		illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[cluster_offset]	  = 0;
		illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[cluster_offset]		  = {};
		illumination_aware_kd_tree.learning_to_cluster.light_cluster_batch_statistics[cluster_offset] = {};
	}
}

#endif
