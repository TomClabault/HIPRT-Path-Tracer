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
inline IlluminationAwareKDTree_CorePromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
													   unsigned long long int original_guiding_node_count,
													   int x)
#else  // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CorePromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int guiding_list_index	  = blockIdx.x;
	unsigned int thread_slot		  = threadIdx.x;
	unsigned int active_guiding_count = *illumination_aware_kd_tree.core.active_guiding_node_count;
	if (guiding_list_index >= active_guiding_count)
		return;
#else  // #ifdef __KERNELCC__
	unsigned int guiding_list_index = static_cast<unsigned int>(x);
	unsigned int thread_slot		= 0;
	if (guiding_list_index >= original_guiding_node_count)
		return;
#endif // #ifdef __KERNELCC__

	if (illumination_aware_kd_tree.core.needs_split[guiding_list_index] == 0)
		return;

	unsigned int parent_index = illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index];
	unsigned int node_count	  = *illumination_aware_kd_tree.core.node_count;
	if (parent_index >= node_count)
		return;

	IlluminationAwareKDTreeNode& parent = illumination_aware_kd_tree.core.nodes[parent_index];
	if (!(parent.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
		return;

	unsigned int left_child_index  = parent.left_child_index;
	unsigned int right_child_index = left_child_index + 1u;
	unsigned int parent_set_index  = parent.lightcut_normal_set_index;

	IlluminationAwareKDTreeNode& left_child	 = illumination_aware_kd_tree.core.nodes[left_child_index];
	IlluminationAwareKDTreeNode& right_child = illumination_aware_kd_tree.core.nodes[right_child_index];

#ifdef __KERNELCC__
	__shared__ unsigned int right_set_index;
	__shared__ unsigned int active_guiding_output_index;
	__shared__ bool active_guiding_allocation_valid;
	__shared__ bool right_set_allocation_valid;
	__shared__ IlluminationAwareKDTreeLearningToClusterLightcutSet right_set;

	if (thread_slot == 0)
	{
		right_set_index					= hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_set_count, 1u);
		active_guiding_output_index		= hippt::atomic_fetch_add(illumination_aware_kd_tree.core.active_guiding_node_count, 1u);
		active_guiding_allocation_valid = active_guiding_output_index < illumination_aware_kd_tree.core.node_capacity;
		right_set_allocation_valid		= right_set_index < illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_set_capacity;
		right_set.initialize_invalid();
	}

	__syncthreads();
	if (!active_guiding_allocation_valid)
		return;

	if (thread_slot == 0)
	{
		left_child.lightcut_normal_set_index  = parent_set_index;
		right_child.lightcut_normal_set_index = right_set_allocation_valid ? right_set_index : IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

		left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
		right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;

		parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;
		parent.lightcut_normal_set_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

		illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(left_child_index, illumination_aware_kd_tree.core.node_capacity);
		illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(right_child_index, illumination_aware_kd_tree.core.node_capacity);

		illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index]		  = left_child_index;
		illumination_aware_kd_tree.core.active_guiding_nodes[active_guiding_output_index] = right_child_index;
	}

	__syncthreads();

	if (right_set_allocation_valid &&
		thread_slot < SurfaceNormalFace_Count * IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT)
	{
		unsigned int normal_face		   = thread_slot / IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
		unsigned int lightcut_variant	   = thread_slot % IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
		unsigned int parent_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
		if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		{
			const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& parent_face =
				illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face];
			if (lightcut_variant == 0u)
				parent_lightcut_index = parent_face.shared_lightcut_index;
			else
				parent_lightcut_index = parent_face.per_mesh_id_lightcuts[lightcut_variant - 1u].lightcut_index;
		}

		if (parent_lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		{
			unsigned int right_lightcut_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.lightcut_count, 1u);

			if (right_lightcut_index < illumination_aware_kd_tree.learning_to_cluster.lightcut_capacity)
			{
				if (lightcut_variant == 0u)
					right_set.face_lightcuts[normal_face].shared_lightcut_index = right_lightcut_index;
				else
				{
					const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& parent_face =
						illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face];
					right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[lightcut_variant - 1u].mesh_id =
						parent_face.per_mesh_id_lightcuts[lightcut_variant - 1u].mesh_id;
					right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[lightcut_variant - 1u].lightcut_index = right_lightcut_index;
				}
				hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
			}
		}
	}

	__syncthreads();

	if (right_set_allocation_valid && parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX && thread_slot == 0)
	{
		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& parent_face =
				illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face];
			for (unsigned int lightcut_variant = 0; lightcut_variant < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
				 lightcut_variant++)
			{
				unsigned int parent_lightcut_index =
					lightcut_variant == 0u ? parent_face.shared_lightcut_index : parent_face.per_mesh_id_lightcuts[lightcut_variant - 1u].lightcut_index;
				unsigned int right_lightcut_index = lightcut_variant == 0u
														? right_set.face_lightcuts[normal_face].shared_lightcut_index
														: right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[lightcut_variant - 1u].lightcut_index;

				if (parent_lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX &&
					right_lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					illumination_aware_kd_tree.learning_to_cluster.clone_lightcut_as_fresh_child(parent_lightcut_index, right_lightcut_index);
			}
		}
	}

	__syncthreads();

	if (thread_slot == 0)
	{
		if (right_set_allocation_valid)
		{
			illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[right_set_index] = right_set;

			for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
			{
				unsigned int parent_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
				if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					parent_lightcut_index =
						illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face].shared_lightcut_index;
				const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts* parent_face = nullptr;
				if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					parent_face = &illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face];
				if (parent_face == nullptr)
				{
					unsigned int right_observation_offset =
						illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);
					illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset] = 0u;
					continue;
				}

				unsigned int parent_observation_offset =
					illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(parent_set_index, normal_face);
				unsigned int right_observation_offset =
					illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);

				unsigned int right_lightcut_index = right_set.face_lightcuts[normal_face].shared_lightcut_index;
				if (parent_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX ||
					right_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
				{
					illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset] = 0;
					illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset]	 = 0;

					continue;
				}

				for (unsigned int lightcut_variant = 0; lightcut_variant < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
					 lightcut_variant++)
				{
					unsigned int parent_variant_lightcut_index =
						lightcut_variant == 0u ? parent_face->shared_lightcut_index : parent_face->per_mesh_id_lightcuts[lightcut_variant - 1u].lightcut_index;
					if (parent_variant_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
						continue;

					for (unsigned int slot = 0; slot < LearningToClusterMaximumLightCutSize; slot++)
					{
						unsigned int parent_offset =
							illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(parent_variant_lightcut_index, slot);
						float parent_estimated_importance_Q =
							illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[parent_offset].estimated_importance_Q;
						illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[parent_offset] = {};
						illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[parent_offset].initialize_importance_prior(
							parent_estimated_importance_Q);
						illumination_aware_kd_tree.learning_to_cluster.lightcut_batch_statistics.reset(parent_offset);
					}

					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_variant_lightcut_index].iteration				  = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_variant_lightcut_index].last_refinement_iteration = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_variant_lightcut_index].refinement_stopped		  = false;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[parent_variant_lightcut_index]				  = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[parent_variant_lightcut_index] =
						IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
				}

				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index] =
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index];

				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index].iteration				 = 0;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index].last_refinement_iteration = 0;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index].refinement_stopped		 = false;

				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index].iteration				  = 0;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index].last_refinement_iteration = 0;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index].refinement_stopped		  = false;

				illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[right_lightcut_index]	 = 0;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[parent_lightcut_index] = 0;

				illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[right_lightcut_index] =
					IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[parent_lightcut_index] =
					IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

				for (unsigned int per_mesh_id_lightcut_slot = 0;
					 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
				{
					unsigned int parent_per_mesh_id_lightcut_index = parent_face->per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index;
					unsigned int right_per_mesh_id_lightcut_index =
						right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index;
					if (parent_per_mesh_id_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX ||
						right_per_mesh_id_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
						continue;

					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index] =
						illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_per_mesh_id_lightcut_index];
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index].iteration				 = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index].last_refinement_iteration = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index].refinement_stopped		 = false;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[right_per_mesh_id_lightcut_index]					 = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[right_per_mesh_id_lightcut_index] =
						IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[parent_per_mesh_id_lightcut_index] = 0u;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[parent_per_mesh_id_lightcut_index] =
						IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
				}

				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset]	 = 0u;
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset] = 0u;
			}
		}
	}

	__syncthreads();

#else  // #ifdef __KERNELCC__

	unsigned int right_set_index			 = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_set_count, 1u);
	unsigned int active_guiding_output_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.core.active_guiding_node_count, 1u);
	if (active_guiding_output_index >= illumination_aware_kd_tree.core.node_capacity)
		return;

	bool right_set_allocation_valid = right_set_index < illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_set_capacity;
	IlluminationAwareKDTreeLearningToClusterLightcutSet right_set{};
	right_set.initialize_invalid();

	left_child.lightcut_normal_set_index  = parent_set_index;
	right_child.lightcut_normal_set_index = right_set_allocation_valid ? right_set_index : IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
	left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
	left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
	right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
	right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
	parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;
	parent.lightcut_normal_set_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

	illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(left_child_index, illumination_aware_kd_tree.core.node_capacity);
	illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(right_child_index, illumination_aware_kd_tree.core.node_capacity);
	illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index]		  = left_child_index;
	illumination_aware_kd_tree.core.active_guiding_nodes[active_guiding_output_index] = right_child_index;

	if (right_set_allocation_valid && parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
	{
		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			unsigned int parent_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
			if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
				parent_lightcut_index =
					illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face].shared_lightcut_index;

			if (parent_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
				continue;

			unsigned int right_lightcut_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.lightcut_count, 1u);
			if (right_lightcut_index >= illumination_aware_kd_tree.learning_to_cluster.lightcut_capacity)
				continue;

			right_set.face_lightcuts[normal_face].shared_lightcut_index = right_lightcut_index;
			// Inherit the estimated importance as the prior while starting fresh Q and refinement observations in both new cells.
			illumination_aware_kd_tree.learning_to_cluster.clone_lightcut_as_fresh_child(parent_lightcut_index, right_lightcut_index);
			hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
		}

		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& parent_face =
				illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face];
			for (unsigned int per_mesh_id_lightcut_slot = 0;
				 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
			{
				unsigned int parent_lightcut_index = parent_face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index;
				if (parent_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					continue;

				unsigned int right_lightcut_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.lightcut_count, 1u);
				if (right_lightcut_index >= illumination_aware_kd_tree.learning_to_cluster.lightcut_capacity)
					continue;

				right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id =
					parent_face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id;
				right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index = right_lightcut_index;
				illumination_aware_kd_tree.learning_to_cluster.clone_lightcut_as_fresh_child(parent_lightcut_index, right_lightcut_index);
				hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
			}
		}

		illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[right_set_index] = right_set;

		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			unsigned int parent_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
			if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
				parent_lightcut_index =
					illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face].shared_lightcut_index;

			unsigned int right_lightcut_index = right_set.face_lightcuts[normal_face].shared_lightcut_index;
			unsigned int parent_observation_offset =
				illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(parent_set_index, normal_face);
			unsigned int right_observation_offset =
				illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);

			if (parent_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
			{
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset] = 0;
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset]	 = 0;

				continue;
			}

			unsigned int right_lightcut_data_index = right_lightcut_index;
			if (right_lightcut_data_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
				continue;

			const IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& parent_face =
				illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[parent_set_index].face_lightcuts[normal_face];
			for (unsigned int lightcut_variant = 0; lightcut_variant < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
				 lightcut_variant++)
			{
				unsigned int parent_variant_lightcut_index =
					lightcut_variant == 0u ? parent_face.shared_lightcut_index : parent_face.per_mesh_id_lightcuts[lightcut_variant - 1u].lightcut_index;
				if (parent_variant_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					continue;

				for (unsigned int slot = 0; slot < LearningToClusterMaximumLightCutSize; slot++)
				{
					unsigned int parent_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(parent_variant_lightcut_index, slot);
					float parent_estimated_importance_Q =
						illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[parent_offset].estimated_importance_Q;
					illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[parent_offset] = {};
					illumination_aware_kd_tree.learning_to_cluster.lightcut_statistics[parent_offset].initialize_importance_prior(
						parent_estimated_importance_Q);
					illumination_aware_kd_tree.learning_to_cluster.lightcut_batch_statistics.reset(parent_offset);
				}

				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_variant_lightcut_index].iteration				  = 0u;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_variant_lightcut_index].last_refinement_iteration = 0u;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_variant_lightcut_index].refinement_stopped		  = false;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[parent_variant_lightcut_index]				  = 0u;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[parent_variant_lightcut_index] =
					IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
			}

			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_data_index] =
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index];
			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index].iteration				 = 0;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index].last_refinement_iteration = 0;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_lightcut_index].refinement_stopped		 = false;

			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index].iteration				  = 0;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index].last_refinement_iteration = 0;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_lightcut_index].refinement_stopped		  = false;

			illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[right_lightcut_data_index] = 0;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[right_lightcut_data_index] =
				IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

			for (unsigned int per_mesh_id_lightcut_slot = 0;
				 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
			{
				unsigned int parent_per_mesh_id_lightcut_index = parent_face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index;
				unsigned int right_per_mesh_id_lightcut_index =
					right_set.face_lightcuts[normal_face].per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index;
				if (parent_per_mesh_id_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX ||
					right_per_mesh_id_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					continue;

				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index] =
					illumination_aware_kd_tree.learning_to_cluster.lightcut_data[parent_per_mesh_id_lightcut_index];
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index].iteration				 = 0u;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index].last_refinement_iteration = 0u;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_data[right_per_mesh_id_lightcut_index].refinement_stopped		 = false;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[right_per_mesh_id_lightcut_index]					 = 0u;
				illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[right_per_mesh_id_lightcut_index] =
					IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
			}

			illumination_aware_kd_tree.learning_to_cluster.lightcut_sample_counts[parent_lightcut_index] = 0;
			illumination_aware_kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[parent_lightcut_index] =
				IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

			hippt::atomic_exchange(&illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset], 0u);
			hippt::atomic_exchange(&illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset], 0u);
		}
	}

	if (right_set_allocation_valid)
	{
		illumination_aware_kd_tree.learning_to_cluster.normal_lightcut_sets[right_set_index] = right_set;
		if (parent_set_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		{
			for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
			{
				unsigned int right_observation_offset =
					illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset] = 0;
			}
		}
	}
#endif // #ifdef __KERNELCC__

#ifdef __KERNELCC__
	// The promoted subtree starts a fresh illumination-signature-history.
	if (thread_slot == 0)
	{
		__shared__ unsigned int stack[128];
		unsigned int stack_size = 0;
		stack[stack_size++]		= left_child_index;
		stack[stack_size++]		= right_child_index;

		while (stack_size > 0)
		{
			unsigned int node_index										   = stack[--stack_size];
			illumination_aware_kd_tree.core.history_signatures[node_index] = illumination_aware_kd_tree.core.batch_signatures[node_index];

			const IlluminationAwareKDTreeNode& node = illumination_aware_kd_tree.core.nodes[node_index];
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
#else  // #ifdef __KERNELCC__
	unsigned int stack[128];
	unsigned int stack_size = 0;
	stack[stack_size++]		= left_child_index;
	stack[stack_size++]		= right_child_index;
	while (stack_size > 0)
	{
		unsigned int node_index										   = stack[--stack_size];
		illumination_aware_kd_tree.core.history_signatures[node_index] = illumination_aware_kd_tree.core.batch_signatures[node_index];
		const IlluminationAwareKDTreeNode& node						   = illumination_aware_kd_tree.core.nodes[node_index];
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
#endif // #ifdef __KERNELCC__
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_PROMOTE_GUIDING_CELLS_H
