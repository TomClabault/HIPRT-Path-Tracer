/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHTCUTS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHTCUTS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterInitializeLightClustering.h"

#define MinimumNormalFaceObservations 1

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterAllocateNormalFaceLightcuts(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterAllocateNormalFaceLightcuts(IlluminationAwareKDTreeDevice kd_tree)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int slot							= threadIdx.x;
	unsigned int active_guiding_node_face_index = blockIdx.x;
#else
	unsigned int slot							= 0;
	unsigned int active_guiding_node_face_index = static_cast<unsigned int>(x);
#endif // #ifdef __KERNELCC__

	unsigned int active_guiding_count = *kd_tree.core.active_guiding_node_count;
	if (active_guiding_node_face_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_guiding_node_face_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_guiding_node_face_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.core.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.core.nodes[guiding_node_index].lightcut_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		return;

	IlluminationAwareKDTreeLearningToClusterLightcutSet::PerNormalFaceLightcuts& face =
		kd_tree.learning_to_cluster.normal_lightcut_sets[set_index].face_lightcuts[normal_face];
	unsigned int observation_offset = kd_tree.learning_to_cluster.get_normal_face_observation_offset(set_index, normal_face);

#ifdef __KERNELCC__
	__shared__ unsigned int new_lightcut_indices[IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT];
	__shared__ bool allocation_valid[IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT];
	__shared__ bool initialize_from_initial_cut[IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT];
	__shared__ bool clone_from_shared_cut[IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT];

	if (slot == 0)
	{
		for (unsigned int lightcut_variant = 0; lightcut_variant < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
			 lightcut_variant++)
		{
			new_lightcut_indices[lightcut_variant]		  = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
			allocation_valid[lightcut_variant]			  = false;
			initialize_from_initial_cut[lightcut_variant] = false;
			clone_from_shared_cut[lightcut_variant]		  = false;
		}

		new_lightcut_indices[0] = face.shared_lightcut_index;
		if (face.shared_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX &&
			kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] >= MinimumNormalFaceObservations)
		{
			new_lightcut_indices[0]		   = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_count, 1u);
			allocation_valid[0]			   = new_lightcut_indices[0] < kd_tree.learning_to_cluster.lightcut_capacity;
			initialize_from_initial_cut[0] = allocation_valid[0];
		}

		if (new_lightcut_indices[0] != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX &&
			new_lightcut_indices[0] < kd_tree.learning_to_cluster.lightcut_capacity)
		{
			for (unsigned int per_mesh_id_lightcut_slot = 0;
				 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
			{
				unsigned int lightcut_variant = per_mesh_id_lightcut_slot + 1u;
				if (face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id == IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID ||
					face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
					continue;

				new_lightcut_indices[lightcut_variant]	= hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_count, 1u);
				allocation_valid[lightcut_variant]		= new_lightcut_indices[lightcut_variant] < kd_tree.learning_to_cluster.lightcut_capacity;
				clone_from_shared_cut[lightcut_variant] = allocation_valid[lightcut_variant];
			}
		}
	}

	__syncthreads();
	for (unsigned int lightcut_variant = 0; lightcut_variant < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_FACE_NORMAL_LIGHTCUT_COUNT;
		 lightcut_variant++)
	{
		if (initialize_from_initial_cut[lightcut_variant])
			learning_to_cluster_initialize_light_clustering_from_initial_cut(kd_tree, new_lightcut_indices[lightcut_variant], slot);
		else if (clone_from_shared_cut[lightcut_variant] && slot == 0)
			kd_tree.learning_to_cluster.clone_lightcut_as_fresh_child(new_lightcut_indices[0], new_lightcut_indices[lightcut_variant]);

		__syncthreads();
	}

	if (slot == 0)
	{
		if (allocation_valid[0])
		{
			face.shared_lightcut_index = new_lightcut_indices[0];
			hippt::atomic_fetch_add(kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
		}

		for (unsigned int per_mesh_id_lightcut_slot = 0;
			 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
		{
			unsigned int lightcut_variant = per_mesh_id_lightcut_slot + 1u;
			if (allocation_valid[lightcut_variant])
			{
				face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index = new_lightcut_indices[lightcut_variant];
				hippt::atomic_fetch_add(kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
			}
		}
	}
#else  // #ifdef __KERNELCC__
	if (face.shared_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX &&
		kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] >= MinimumNormalFaceObservations)
	{
		unsigned int new_lightcut_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_count, 1u);
		if (new_lightcut_index < kd_tree.learning_to_cluster.lightcut_capacity)
		{
			for (unsigned int lightcut_slot = 0; lightcut_slot < LearningToClusterMaximumLightCutSize; lightcut_slot++)
				learning_to_cluster_initialize_light_clustering_from_initial_cut(kd_tree, new_lightcut_index, lightcut_slot);

			face.shared_lightcut_index = new_lightcut_index;
			hippt::atomic_fetch_add(kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
		}
	}

	if (face.shared_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		return;

	for (unsigned int per_mesh_id_lightcut_slot = 0;
		 per_mesh_id_lightcut_slot < IlluminationAwareKDTreeLearningToClusterLightcutSet::PER_MESH_ID_LIGHTCUT_COUNT; per_mesh_id_lightcut_slot++)
	{
		if (face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].mesh_id == IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID ||
			face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
			continue;

		unsigned int new_lightcut_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_count, 1u);
		if (new_lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
			continue;

		kd_tree.learning_to_cluster.clone_lightcut_as_fresh_child(face.shared_lightcut_index, new_lightcut_index);
		face.per_mesh_id_lightcuts[per_mesh_id_lightcut_slot].lightcut_index = new_lightcut_index;
		hippt::atomic_fetch_add(kd_tree.learning_to_cluster.allocated_lightcut_count, 1u);
	}
#endif // #ifdef __KERNELCC__
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHTCUTS_H
