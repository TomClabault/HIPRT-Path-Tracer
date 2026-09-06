/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REDUCE_BATCH_STATISTICS_UPWARD_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REDUCE_BATCH_STATISTICS_UPWARD_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

/*
 * Batch training samples are accumulated at the deepest existing lookahead node before this kernel is run. A naive approach updates every node visited by every
 * sample. Each visit can require six signature atomics (observation count, radiance sum, squared radiance sum, and three directional components) plus seven
 * spatial moment atomics for positive-radiance samples (count, three position sums, and three squared-position sums). A sample traversing the guiding cell and
 * four lookahead levels could issue up to 5 * 13 = 65 atomics, while six lookahead levels could issue up to 7 * 13 = 91 atomics.
 *
 * This kernel, for each sample, performs its atomics once at the deepest applicable
 * node. This kernel then propagates the node's already-aggregated batch statistics to its parent. The host launches
 * one pass for each reduction level in descending order, so a node receives contributions from deeper nodes before
 * its own statistics are forwarded farther upward. The work is therefore approximately one sample update plus one
 * node-level update per populated tree edge, instead of one update for every sample at every visited level.
 *
 * For example, consider a guiding node G with lookahead descendants A, B, and C, where C is the terminal node for
 * a sample. The sample first updates C. The reduction passes then perform C -> B, B -> A, and A -> G. If a path ends
 * early because a lookahead child does not exist, only the existing edges are reduced. A node is reduced only when
 * it is exactly the requested number of edges below its nearest guiding ancestor, which prevents statistics from
 * crossing a guiding-cell boundary after a promotion. Parent indices are stored explicitly because nodes only
 * store their child links.
 */
#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_CoreReduceBatchStatisticsUpward(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int reduction_level, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CoreReduceBatchStatisticsUpward(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int reduction_level)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int node_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int node_index = static_cast<unsigned int>(x);
#endif

	unsigned int node_count = *kd_tree_device.core.node_count;
	if (node_index >= node_count || reduction_level == 0u)
		return;

	// A node is reduced only when it is exactly reduction_level edges below a guiding cell.
	// This gives the host one globally ordered pass per level without requiring a depth field in every node.
	unsigned int ancestor_index = node_index;
	for (unsigned int distance = 0; distance < reduction_level; distance++)
	{
		if (kd_tree_device.core.nodes[ancestor_index].flags & IlluminationAwareKDTreeNodeFlag_Guiding)
			return;

		unsigned int parent_index = kd_tree_device.core.parent_indices[ancestor_index];
		if (parent_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || parent_index >= node_count)
			return;

		ancestor_index = parent_index;
	}

	if (!(kd_tree_device.core.nodes[ancestor_index].flags & IlluminationAwareKDTreeNodeFlag_Guiding))
		return;

	unsigned int parent_index = kd_tree_device.core.parent_indices[node_index];
	if (parent_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || parent_index >= node_count)
		return;

	const IlluminationAwareKDTreeIlluminationSignature& source_signature = kd_tree_device.core.batch_signatures[node_index];
	if (source_signature.valid_observation_count == 0u)
		return;

	hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_signatures[parent_index].valid_observation_count, source_signature.valid_observation_count);
	hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_signatures[parent_index].scalar_radiance_sum, source_signature.scalar_radiance_sum);
	hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_signatures[parent_index].squared_scalar_radiance_sum, source_signature.squared_scalar_radiance_sum);
	hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_signatures[parent_index].weighted_direction_sum.x, source_signature.weighted_direction_sum.x);
	hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_signatures[parent_index].weighted_direction_sum.y, source_signature.weighted_direction_sum.y);
	hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_signatures[parent_index].weighted_direction_sum.z, source_signature.weighted_direction_sum.z);

	const IlluminationAwareKDTreeSpatialSampleMoments& source_spatial = kd_tree_device.core.batch_spatial_moments[node_index];
	if (source_spatial.positive_radiance_sample_count > 0u)
	{
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].positive_radiance_sample_count,
									source_spatial.positive_radiance_sample_count);
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].position_sum.x, source_spatial.position_sum.x);
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].position_sum.y, source_spatial.position_sum.y);
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].position_sum.z, source_spatial.position_sum.z);
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].position_squared_sum.x, source_spatial.position_squared_sum.x);
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].position_squared_sum.y, source_spatial.position_squared_sum.y);
		hippt::atomic_fetch_add_gpu(&kd_tree_device.core.batch_spatial_moments[parent_index].position_squared_sum.z, source_spatial.position_squared_sum.z);
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REDUCE_BATCH_STATISTICS_UPWARD_H
