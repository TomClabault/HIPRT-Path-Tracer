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

	if (slot >= LearningToClusterLightClusteringBlockSize)
		return;

	if (slot == 0)
	{
		kd_tree.core.nodes[0].light_clustering_normal_set_index	 = 0;
		*kd_tree.learning_to_cluster.light_clustering_count		 = 0;
		*kd_tree.learning_to_cluster.normal_clustering_set_count = 1;
	}

	if (slot < SurfaceNormalFace_Count)
		kd_tree.learning_to_cluster.normal_clustering_sets[0].clustering_indices[slot] = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
}

#endif
