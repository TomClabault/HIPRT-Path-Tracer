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
inline IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
												   unsigned long long int original_guiding_node_count,
												   int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_PromoteGuidingCells(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif
{
#ifdef __KERNELCC__
	unsigned int guiding_list_index	  = blockIdx.x;
	unsigned int thread_slot		  = threadIdx.x;
	unsigned int active_guiding_count = *illumination_aware_kd_tree.core.active_guiding_node_count;
	if (guiding_list_index >= active_guiding_count)
		return;
#else
	unsigned int guiding_list_index = static_cast<unsigned int>(x);
	unsigned int thread_slot		= 0;
	if (guiding_list_index >= original_guiding_node_count)
		return;
#endif

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
	unsigned int parent_set_index  = parent.light_clustering_normal_set_index;

	IlluminationAwareKDTreeNode& left_child	 = illumination_aware_kd_tree.core.nodes[left_child_index];
	IlluminationAwareKDTreeNode& right_child = illumination_aware_kd_tree.core.nodes[right_child_index];

#ifdef __KERNELCC__
	__shared__ unsigned int right_set_index;
	__shared__ unsigned int active_guiding_output_index;
	__shared__ bool active_guiding_allocation_valid;
	__shared__ bool right_set_allocation_valid;
	__shared__ IlluminationAwareKDTreeNormalClusteringSet right_set;

	if (thread_slot == 0)
	{
		right_set_index					= hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.normal_clustering_set_count, 1u);
		active_guiding_output_index		= hippt::atomic_fetch_add(illumination_aware_kd_tree.core.active_guiding_node_count, 1u);
		active_guiding_allocation_valid = active_guiding_output_index < illumination_aware_kd_tree.core.node_capacity;
		right_set_allocation_valid		= right_set_index < illumination_aware_kd_tree.learning_to_cluster.normal_clustering_set_capacity;

		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
			right_set.clustering_indices[normal_face] = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
	}

	__syncthreads();
	if (!active_guiding_allocation_valid)
		return;

	if (thread_slot == 0)
	{
		left_child.light_clustering_normal_set_index = parent_set_index;
		right_child.light_clustering_normal_set_index =
			right_set_allocation_valid ? right_set_index : IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

		left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
		right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
		right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;

		parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;
		parent.light_clustering_normal_set_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

		illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(left_child_index, illumination_aware_kd_tree.core.node_capacity);
		illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(right_child_index, illumination_aware_kd_tree.core.node_capacity);

		illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index]		  = left_child_index;
		illumination_aware_kd_tree.core.active_guiding_nodes[active_guiding_output_index] = right_child_index;
	}

	__syncthreads();

	if (right_set_allocation_valid && thread_slot < SurfaceNormalFace_Count)
	{
		unsigned int parent_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
		if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
			parent_clustering_index = illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[parent_set_index].clustering_indices[thread_slot];

		if (parent_clustering_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		{
			unsigned int right_clustering_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.light_clustering_count, 1u);

			if (right_clustering_index < illumination_aware_kd_tree.learning_to_cluster.light_clustering_capacity)
				right_set.clustering_indices[thread_slot] = right_clustering_index;
		}
	}

	__syncthreads();

	if (right_set_allocation_valid && parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
	{
		for (unsigned int slot_iteration = 0; slot_iteration < LearningToClusterMaximumLightCutSize; slot_iteration++)
		{
			unsigned int slot = thread_slot + slot_iteration;
			if (slot >= LearningToClusterMaximumLightCutSize)
				continue;

			for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
			{
				unsigned int parent_clustering_index =
					illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[parent_set_index].clustering_indices[normal_face];
				unsigned int right_clustering_index = right_set.clustering_indices[normal_face];

				if (parent_clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX ||
					right_clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
					continue;

				unsigned int source_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(parent_clustering_index, slot);
				unsigned int right_offset  = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(right_clustering_index, slot);

				illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[right_offset] =
					illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[source_offset];
				illumination_aware_kd_tree.learning_to_cluster.light_cluster_cdfs[right_offset] =
					illumination_aware_kd_tree.learning_to_cluster.light_cluster_cdfs[source_offset];
				illumination_aware_kd_tree.learning_to_cluster.reservoir_proposals[right_offset] = 0ull;

				// Only inheriting the estimated importance Q, not the other statistics, because we want to start learning fresh for both the new cells
				float parent_estimated_importance_Q =
					illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[source_offset].estimated_importance_Q;

				illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[source_offset] = IlluminationAwareKDTreeLightClusterStatistics{};
				illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[right_offset]  = IlluminationAwareKDTreeLightClusterStatistics{};

				illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[source_offset].estimated_importance_Q = parent_estimated_importance_Q;
				illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[right_offset].estimated_importance_Q  = parent_estimated_importance_Q;
			}
		}
	}

	__syncthreads();

	if (thread_slot == 0)
	{
		if (right_set_allocation_valid)
		{
			illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[right_set_index] = right_set;

			for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
			{
				unsigned int parent_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
				if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
					parent_clustering_index =
						illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[parent_set_index].clustering_indices[normal_face];

				unsigned int parent_observation_offset =
					illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(parent_set_index, normal_face);
				unsigned int right_observation_offset =
					illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);

				unsigned int right_clustering_index = right_set.clustering_indices[normal_face];
				if (parent_clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX ||
					right_clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
				{
					illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset] = 0;
					illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset]	 = 0;

					continue;
				}

				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index] =
					illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index];

				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].pending_record_budget	   = 0;
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].iteration				   = 0;
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].last_refinement_iteration = 0;
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].refinement_stopped		   = false;

				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].pending_record_budget		= 0;
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].iteration					= 0;
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].last_refinement_iteration = 0;
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].refinement_stopped		= false;

				illumination_aware_kd_tree.learning_to_cluster.pending_light_cluster_record_counts[right_clustering_index]	= 0;
				illumination_aware_kd_tree.learning_to_cluster.pending_light_cluster_record_counts[parent_clustering_index] = 0;

				illumination_aware_kd_tree.learning_to_cluster.reservoir_seen_counts[right_clustering_index]  = 0;
				illumination_aware_kd_tree.learning_to_cluster.reservoir_seen_counts[parent_clustering_index] = 0;

				illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states[right_clustering_index] =
					IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;
				illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states[parent_clustering_index] =
					IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset]	 = 0u;
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset] = 0u;
			}
		}
	}

	__syncthreads();

#else

	unsigned int right_set_index			 = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.normal_clustering_set_count, 1u);
	unsigned int active_guiding_output_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.core.active_guiding_node_count, 1u);
	if (active_guiding_output_index >= illumination_aware_kd_tree.core.node_capacity)
		return;

	bool right_set_allocation_valid = right_set_index < illumination_aware_kd_tree.learning_to_cluster.normal_clustering_set_capacity;
	IlluminationAwareKDTreeNormalClusteringSet right_set{};
	for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		right_set.clustering_indices[normal_face] = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

	left_child.light_clustering_normal_set_index  = parent_set_index;
	right_child.light_clustering_normal_set_index = right_set_allocation_valid ? right_set_index : IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
	left_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
	left_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
	right_child.flags |= IlluminationAwareKDTreeNodeFlag_Guiding;
	right_child.flags &= ~IlluminationAwareKDTreeNodeFlag_Lookahead;
	parent.flags &= ~IlluminationAwareKDTreeNodeFlag_Guiding;
	parent.light_clustering_normal_set_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;

	illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(left_child_index, illumination_aware_kd_tree.core.node_capacity);
	illumination_aware_kd_tree.nisml.initialize_nisml_cache_for_guiding_cell(right_child_index, illumination_aware_kd_tree.core.node_capacity);
	illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index]		  = left_child_index;
	illumination_aware_kd_tree.core.active_guiding_nodes[active_guiding_output_index] = right_child_index;

	if (right_set_allocation_valid && parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
	{
		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			unsigned int parent_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
			if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
				parent_clustering_index =
					illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[parent_set_index].clustering_indices[normal_face];

			if (parent_clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
				continue;

			unsigned int right_clustering_index = hippt::atomic_fetch_add(illumination_aware_kd_tree.learning_to_cluster.light_clustering_count, 1u);
			if (right_clustering_index >= illumination_aware_kd_tree.learning_to_cluster.light_clustering_capacity)
				continue;

			right_set.clustering_indices[normal_face] = right_clustering_index;
			for (unsigned int slot = 0; slot < LearningToClusterMaximumLightCutSize; slot++)
			{
				unsigned int source_offset = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(parent_clustering_index, slot);
				unsigned int right_offset  = illumination_aware_kd_tree.learning_to_cluster.get_light_cluster_offset(right_clustering_index, slot);

				illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[right_offset] =
					illumination_aware_kd_tree.learning_to_cluster.light_cluster_node_indices[source_offset];
				illumination_aware_kd_tree.learning_to_cluster.light_cluster_cdfs[right_offset] =
					illumination_aware_kd_tree.learning_to_cluster.light_cluster_cdfs[source_offset];
				illumination_aware_kd_tree.learning_to_cluster.reservoir_proposals[right_offset] = 0ull;
			}

			// Only inheriting the estimated importance Q, not the other statistics, because we want to start learning fresh for both the new cells
			float parent_estimated_importance_Q =
				illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[parent_clustering_index].estimated_importance_Q;

			illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[right_clustering_index] = IlluminationAwareKDTreeLightClusterStatistics{};
			illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[right_clustering_index].estimated_importance_Q =
				parent_estimated_importance_Q;

			illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[parent_clustering_index] = IlluminationAwareKDTreeLightClusterStatistics{};
			illumination_aware_kd_tree.learning_to_cluster.light_cluster_statistics[parent_clustering_index].estimated_importance_Q =
				parent_estimated_importance_Q;
		}

		illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[right_set_index] = right_set;

		for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
		{
			unsigned int parent_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
			if (parent_set_index != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
				parent_clustering_index =
					illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[parent_set_index].clustering_indices[normal_face];

			unsigned int right_clustering_index = right_set.clustering_indices[normal_face];
			unsigned int parent_observation_offset =
				illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(parent_set_index, normal_face);
			unsigned int right_observation_offset =
				illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);

			if (parent_clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
			{
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset] = 0;
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset]	 = 0;

				continue;
			}

			unsigned int right_clustering_data_index = right_clustering_index;
			if (right_clustering_data_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
				continue;

			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_data_index] =
				illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index];
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].pending_record_budget	   = 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].iteration				   = 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].last_refinement_iteration = 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_index].refinement_stopped		   = false;

			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].pending_record_budget		= 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].iteration					= 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].last_refinement_iteration = 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[parent_clustering_index].refinement_stopped		= false;

			illumination_aware_kd_tree.learning_to_cluster.pending_light_cluster_record_counts[right_clustering_data_index]			= 0;
			illumination_aware_kd_tree.learning_to_cluster.reservoir_seen_counts[right_clustering_data_index]						= 0;
			illumination_aware_kd_tree.learning_to_cluster.light_clustering_data[right_clustering_data_index].pending_record_budget = 0;
			illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states[right_clustering_data_index] =
				IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

			illumination_aware_kd_tree.learning_to_cluster.pending_light_cluster_record_counts[parent_clustering_index] = 0;
			illumination_aware_kd_tree.learning_to_cluster.reservoir_seen_counts[parent_clustering_index]				= 0;
			illumination_aware_kd_tree.learning_to_cluster.representative_shading_context_states[parent_clustering_index] =
				IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT;

			hippt::atomic_exchange(&illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset], 0u);
			hippt::atomic_exchange(&illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[parent_observation_offset], 0u);
		}
	}

	if (right_set_allocation_valid)
	{
		illumination_aware_kd_tree.learning_to_cluster.normal_clustering_sets[right_set_index] = right_set;
		if (parent_set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		{
			for (unsigned int normal_face = 0; normal_face < SurfaceNormalFace_Count; normal_face++)
			{
				unsigned int right_observation_offset =
					illumination_aware_kd_tree.learning_to_cluster.get_normal_face_observation_offset(right_set_index, normal_face);
				illumination_aware_kd_tree.learning_to_cluster.normal_face_observation_counts[right_observation_offset] = 0;
			}
		}
	}
#endif

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
#else
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
#endif
}

#endif
