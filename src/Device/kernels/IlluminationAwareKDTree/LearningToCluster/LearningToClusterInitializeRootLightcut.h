/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_LIGHTCUT_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_LIGHTCUT_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterInitializeRootLightcut(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterInitializeRootLightcut(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int slot = threadIdx.x;
#else
	unsigned int slot = static_cast<unsigned int>(x);
#endif

	if (slot >= LearningToClusterMaximumLightCutSize)
		return;

	if (slot == 0)
	{
		kd_tree.core.nodes[0].lightcut_normal_set_index		   = 0;
		*kd_tree.learning_to_cluster.lightcut_count			   = 0;
		*kd_tree.learning_to_cluster.allocated_lightcut_count  = 0;
		*kd_tree.learning_to_cluster.normal_lightcut_set_count = 1;
		kd_tree.learning_to_cluster.normal_lightcut_sets[0].initialize_invalid();
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_ROOT_LIGHTCUT_H
