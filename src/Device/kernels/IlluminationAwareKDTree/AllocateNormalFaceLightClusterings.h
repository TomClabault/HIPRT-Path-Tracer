/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ALLOCATE_NORMAL_FACE_LIGHT_CLUSTERINGS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/kernels/IlluminationAwareKDTree/InitializeLightClustering.h"

#define MinimumNormalFaceObservations 1

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_AllocateNormalFaceLightClusterings(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_AllocateNormalFaceLightClusterings(IlluminationAwareKDTreeDevice kd_tree)
#endif
{
#ifdef __KERNELCC__
	unsigned int slot			   = threadIdx.x;
	unsigned int active_pair_index = blockIdx.x;
#else
	unsigned int slot			   = 0;
	unsigned int active_pair_index = static_cast<unsigned int>(x);
#endif

	unsigned int active_guiding_count = *kd_tree.active_guiding_node_count;
	if (active_pair_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_pair_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_pair_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	IlluminationAwareKDTreeNormalClusteringSet& clustering_set = kd_tree.learning_to_cluster.normal_clustering_sets[set_index];
	unsigned int observation_offset							   = kd_tree.learning_to_cluster.get_normal_face_observation_offset(set_index, normal_face);

#ifdef __KERNELCC__
	__shared__ unsigned int new_clustering_index;
	__shared__ bool allocation_valid;

	if (slot == 0)
	{
		new_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
		allocation_valid	 = false;

		if (clustering_set.clustering_indices[normal_face] == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX &&
			kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] >= MinimumNormalFaceObservations)
		{
			new_clustering_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.light_clustering_count, 1u);
			allocation_valid	 = new_clustering_index < kd_tree.learning_to_cluster.light_clustering_capacity;
		}
	}

	__syncthreads();
	if (!allocation_valid)
		return;

	initialize_light_clustering_from_initial_cut(kd_tree, new_clustering_index, slot);
	__syncthreads();

	if (slot == 0)
		clustering_set.clustering_indices[normal_face] = new_clustering_index;
#else
	if (clustering_set.clustering_indices[normal_face] != IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX ||
		kd_tree.learning_to_cluster.normal_face_observation_counts[observation_offset] < MinimumNormalFaceObservations)
		return;

	unsigned int new_clustering_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.light_clustering_count, 1u);
	if (new_clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;

	for (unsigned int cluster_slot = 0; cluster_slot < IlluminationAwareKDTreeMaximumLightCutSize; cluster_slot++)
		initialize_light_clustering_from_initial_cut(kd_tree, new_clustering_index, cluster_slot);

	clustering_set.clustering_indices[normal_face] = new_clustering_index;
#endif
}

#endif
