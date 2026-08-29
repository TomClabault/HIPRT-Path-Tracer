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
inline IlluminationAwareKDTree_CoreResetTree(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
											 const float3_t scene_bounds_minimum,
											 const float3_t scene_bounds_maximum,
											 unsigned int reset_index)
#else // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_CoreResetTree(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
											 const float3 scene_bounds_minimum,
											 const float3 scene_bounds_maximum)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int reset_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int maximum_reset_count = illumination_aware_kd_tree.learning_to_cluster.lightcut_capacity;
	if (illumination_aware_kd_tree.core.node_capacity > maximum_reset_count)
		maximum_reset_count = illumination_aware_kd_tree.core.node_capacity;

	if (reset_index >= maximum_reset_count)
		return;

	if (reset_index < illumination_aware_kd_tree.core.node_capacity)
	{
		if (reset_index == 0)
		{
			IlluminationAwareKDTreeNode root{};
			root.flags					   = IlluminationAwareKDTreeNodeFlag_Guiding;
			root.lightcut_normal_set_index = 0;

			illumination_aware_kd_tree.core.nodes[0]			   = root;
			illumination_aware_kd_tree.core.node_bounds[0].minimum = scene_bounds_minimum;
			illumination_aware_kd_tree.core.node_bounds[0].maximum = scene_bounds_maximum;

			*illumination_aware_kd_tree.core.node_count								  = 1;
			*illumination_aware_kd_tree.learning_to_cluster.lightcut_count			  = 0;
			*illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_set_count = 1;
			*illumination_aware_kd_tree.core.active_guiding_node_count				  = 1;
			illumination_aware_kd_tree.core.active_guiding_nodes[0]					  = 0;

			*illumination_aware_kd_tree.core.training_sample_count = 0;
			if (illumination_aware_kd_tree.nisml.nisml_pending_cell_count != nullptr)
				*illumination_aware_kd_tree.nisml.nisml_pending_cell_count = 0;
			*illumination_aware_kd_tree.learning_to_cluster.training_sample_count = 0;
			*illumination_aware_kd_tree.core.current_frontier_count				  = 0;
			*illumination_aware_kd_tree.core.next_frontier_count				  = 0;
		}

		illumination_aware_kd_tree.core.history_signatures[reset_index]		 = {};
		illumination_aware_kd_tree.core.history_spatial_moments[reset_index] = {};
		illumination_aware_kd_tree.core.batch_signatures[reset_index]		 = {};
		illumination_aware_kd_tree.core.batch_spatial_moments[reset_index]	 = {};
		illumination_aware_kd_tree.core.needs_split[reset_index]			 = 0;
		illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(reset_index, illumination_aware_kd_tree.core.node_capacity);
	}

	if (reset_index < illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_set_capacity)
	{
		IlluminationAwareKDTreeNormalClusteringSet& lightcut_set = illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[reset_index];
		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			lightcut_set.lightcut_indices[normal_face] = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
			unsigned int observation_offset = illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(reset_index, normal_face);
			illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] = 0;
		}
	}

	if (reset_index < illumination_aware_kd_tree.learning_to_cluster.lightcut_capacity)
	{
		illumination_aware_kd_tree.learning_to_cluster.lightcut_data[reset_index]							 = {};
		illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[reset_index]					 = 0;
		illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_contexts[reset_index] = {};
		illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[reset_index] =
			IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

		for (unsigned int slot = 0; slot < LearningToClusterMaximumLightCutSize; slot++)
		{
			unsigned int cluster_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(reset_index, slot);
			illumination_aware_kd_tree.learning_to_cluster.lightcut_node_indices[cluster_offset] = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[cluster_offset]	 = {};
			illumination_aware_kd_tree.learning_to_cluster.lightcut_cdfs[cluster_offset]		 = 0u;
		}
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_NODE_H
