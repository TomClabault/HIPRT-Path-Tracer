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
										 unsigned int reset_index)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ResetTree(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
										 const float3 scene_bounds_minimum,
										 const float3 scene_bounds_maximum)
#endif
{
#ifdef __KERNELCC__
	unsigned int reset_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int maximum_reset_count = illumination_aware_kd_tree.learning_to_cluster.light_clustering_capacity;
	if (illumination_aware_kd_tree.node_capacity > maximum_reset_count)
		maximum_reset_count = illumination_aware_kd_tree.node_capacity;

	if (reset_index >= maximum_reset_count)
		return;

	if (reset_index < illumination_aware_kd_tree.node_capacity)
	{
		if (reset_index == 0)
		{
			IlluminationAwareKDTreeNode root{};
			root.flags							   = IlluminationAwareKDTreeNodeFlag_Guiding;
			root.light_clustering_normal_set_index = 0;

			illumination_aware_kd_tree.nodes[0]				  = root;
			illumination_aware_kd_tree.node_bounds[0].minimum = scene_bounds_minimum;
			illumination_aware_kd_tree.node_bounds[0].maximum = scene_bounds_maximum;

			*illumination_aware_kd_tree.node_count										= 1;
			*illumination_aware_kd_tree.learning_to_cluster.light_clustering_count		= 0;
			*illumination_aware_kd_tree.learning_to_cluster.normal_clustering_set_count = 1;
			*illumination_aware_kd_tree.active_guiding_node_count						= 1;
			illumination_aware_kd_tree.active_guiding_nodes[0]							= 0;

			*illumination_aware_kd_tree.training_sample_count					  = 0;
			*illumination_aware_kd_tree.learning_to_cluster_training_sample_count = 0;
			*illumination_aware_kd_tree.current_frontier_count					  = 0;
			*illumination_aware_kd_tree.next_frontier_count						  = 0;
		}

		illumination_aware_kd_tree.history_signatures.reset(reset_index);
		illumination_aware_kd_tree.history_spatial_moments.reset(reset_index);
		illumination_aware_kd_tree.batch_signatures.reset(reset_index);
		illumination_aware_kd_tree.batch_spatial_moments.reset(reset_index);
		illumination_aware_kd_tree.needs_split[reset_index] = 0;
	}

	if (reset_index < illumination_aware_kd_tree.learning_to_cluster.normal_clustering_set_capacity)
	{
		IlluminationAwareKDTreeNormalClusteringSet& clustering_set = illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[reset_index];
		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			clustering_set.clustering_indices[normal_face] = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
			unsigned int observation_offset = illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(reset_index, normal_face);
			illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] = 0;
		}
	}

	if (reset_index < illumination_aware_kd_tree.learning_to_cluster.light_clustering_capacity)
	{
		illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[reset_index]				= {};
		illumination_aware_kd_tree.learning_to_cluster.pending_light_cluster_record_counts[reset_index] = 0;
		illumination_aware_kd_tree.learning_to_cluster.representative_shading_contexts[reset_index]		= {};
		illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states[reset_index] =
			IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

		for (unsigned int slot = 0; slot < IlluminationAwareKDTreeMaximumLightCutSize; slot++)
		{
			unsigned int cluster_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(reset_index, slot);
			illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[cluster_offset] = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
			illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[cluster_offset]	  = {};
		}
	}
}

#endif
