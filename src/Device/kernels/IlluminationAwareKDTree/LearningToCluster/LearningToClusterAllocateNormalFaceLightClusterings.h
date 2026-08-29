/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterInitializeLightClustering.h"

#define MinimumNormalFaceObservations 1

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterAllocateNormalFaceLightClusterings(IlluminationAwareKDTreeDevice kd_tree, int x)
#else // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterAllocateNormalFaceLightClusterings(IlluminationAwareKDTreeDevice kd_tree)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int slot							= threadIdx.x;
	unsigned int active_guiding_node_face_index = blockIdx.x;
#else // #ifdef __KERNELCC__
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

	IlluminationAwareKDTreeNormalClusteringSet& lightcut_set = kd_tree.learning_to_cluster.normal_lightcut_sets[set_index];
	unsigned int observation_offset							 = kd_tree.learning_to_cluster.get_normal_face_observation_offset(set_index, normal_face);

#ifdef __KERNELCC__
	__shared__ unsigned int new_lightcut_index;
	__shared__ bool allocation_valid;

	if (slot == 0)
	{
		new_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
		allocation_valid   = false;

		if (lightcut_set.lightcut_indices[normal_face] == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX &&
			kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] >= MinimumNormalFaceObservations)
		{
			new_lightcut_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_count, 1u);
			allocation_valid   = new_lightcut_index < kd_tree.learning_to_cluster.lightcut_capacity;
		}
	}

	__syncthreads();
	if (!allocation_valid)
		return;

	learning_to_cluster_initialize_light_clustering_from_initial_cut(kd_tree, new_lightcut_index, slot);
	__syncthreads();

	if (slot == 0)
		lightcut_set.lightcut_indices[normal_face] = new_lightcut_index;
#else // #ifdef __KERNELCC__
	if (lightcut_set.lightcut_indices[normal_face] != IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX ||
		kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] < MinimumNormalFaceObservations)
		return;

	unsigned int new_lightcut_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.lightcut_count, 1u);
	if (new_lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;

	for (unsigned int lightcut_slot = 0; lightcut_slot < LearningToClusterMaximumLightCutSize; lightcut_slot++)
		learning_to_cluster_initialize_light_clustering_from_initial_cut(kd_tree, new_lightcut_index, lightcut_slot);

	lightcut_set.lightcut_indices[normal_face] = new_lightcut_index;
#endif // #ifdef __KERNELCC__
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_H
