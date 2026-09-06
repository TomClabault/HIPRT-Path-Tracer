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
inline IlluminationAwareKDTree_CoreInitializeCreatedNodeHistory(IlluminationAwareKDTreeDevice kd_tree_device,
																unsigned int creation_tag,
																AtomicType<unsigned int>* frontier_count_to_clear,
																int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CoreInitializeCreatedNodeHistory(IlluminationAwareKDTreeDevice kd_tree_device,
														 unsigned int creation_tag,
														 AtomicType<unsigned int>* frontier_count_to_clear)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	const uint32_t node_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	const uint32_t node_index = x;
#endif

	const uint32_t node_count = *kd_tree_device.core.node_count;
	if (node_index < node_count && kd_tree_device.core.nodes[node_index].creation_tag == creation_tag)
	{
		// For newly created nodes, we initialize the history with the current batch values (batch values initialized from the sample replay kernel)
		kd_tree_device.core.history_signatures[node_index]		= kd_tree_device.core.batch_signatures[node_index];
		kd_tree_device.core.history_spatial_moments[node_index] = kd_tree_device.core.batch_spatial_moments[node_index];
	}

	if (node_index == 0)
	{
		// Keep the frontier produced by this iteration intact. Clear the alternate buffer for the next iteration instead.
		*frontier_count_to_clear = 0;
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_CREATED_NODE_HISTORY_KERNEL_H
