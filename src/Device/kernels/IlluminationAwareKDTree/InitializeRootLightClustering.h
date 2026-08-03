/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_LIGHT_CLUSTERING_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_LIGHT_CLUSTERING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_InitializeRootLightClustering(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_InitializeRootLightClustering(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
#endif
{
#ifdef __KERNELCC__
	unsigned int slot = threadIdx.x;
#else
	unsigned int slot = static_cast<unsigned int>(x);
#endif

#ifdef __KERNELCC__
	if (blockIdx.x != 0 || slot >= IlluminationAwareKDTreeMaximumLightCutSize)
#else
	if (slot >= IlluminationAwareKDTreeMaximumLightCutSize)
#endif
		return;

	unsigned int initial_cut_size = IlluminationAwareKDTreeInitialLightCutSize;
	unsigned int clustering_index = 0;
	unsigned int offset			  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);

	if (slot < initial_cut_size)
	{
		unsigned int sg_node_index									   = kd_tree.learning_to_cluster.initial_light_cut_node_indices[slot];
		kd_tree.learning_to_cluster.light_cluster_node_indices[offset] = sg_node_index;

		float fallback_Q = hippt::max(light_tree_sg.nodes[sg_node_index].get_total_power(), 0.0f);

		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		statistics.estimated_importance_Q						  = fallback_Q;
		statistics.estimated_second_moment						  = fallback_Q * fallback_Q;
		statistics.variance										  = 0.0f;
		statistics.visit_count									  = 0;
	}
	else
	{
		kd_tree.learning_to_cluster.light_cluster_node_indices[offset] = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
		kd_tree.learning_to_cluster.light_cluster_statistics[offset]   = {};
	}

	kd_tree.learning_to_cluster.light_cluster_batch_statistics[offset] = {};

	if (slot == 0)
	{
		IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
		cluster_data.cut_size									 = initial_cut_size;
		cluster_data.iteration									 = 0;
		cluster_data.last_refinement_iteration					 = 0;
		cluster_data.refinement_sample_count					 = 0;
		cluster_data.Q0_initialized								 = false;
		cluster_data.refinement_stopped							 = false;

		*kd_tree.learning_to_cluster.light_clustering_batch_sample_counts  = 0;
		*kd_tree.learning_to_cluster.representative_shading_context_states = 0;
	}
}

#endif
