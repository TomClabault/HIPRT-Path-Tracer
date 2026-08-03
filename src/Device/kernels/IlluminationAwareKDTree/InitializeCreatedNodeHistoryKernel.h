/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_CREATED_NODE_HISTORY_KERNEL_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_CREATED_NODE_HISTORY_KERNEL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_InitializeCreatedNodeHistoryKernel(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int creation_tag, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_InitializeCreatedNodeHistoryKernel(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int creation_tag)
#endif
{
#ifdef __KERNELCC__
	unsigned int node_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int node_index = x;
#endif

	unsigned int node_count = *illumination_aware_kd_tree.node_count;
	if (node_index >= node_count)
		return;

	if (illumination_aware_kd_tree.nodes[node_index].creation_tag != creation_tag)
		return;

	// For newly created nodes, we initialize the history with the current batch values (batch values initialized from the sample replay kernel)
	illumination_aware_kd_tree.history_signatures.write(node_index, illumination_aware_kd_tree.batch_signatures.read(node_index));
	illumination_aware_kd_tree.history_spatial_moments.write(node_index, illumination_aware_kd_tree.batch_spatial_moments.read(node_index));
}

#endif
