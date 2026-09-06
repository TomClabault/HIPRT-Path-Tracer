/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_TRAINING_SAMPLES_KERNEL_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_TRAINING_SAMPLES_KERNEL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_CoreReplayTrainingSamples(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int creation_tag, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CoreReplayTrainingSamples(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int creation_tag)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int sample_index = x;
#endif

	unsigned int sample_count = *kd_tree_device.core.training_sample_count;
	if (sample_index >= sample_count)
		return;

	IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample = kd_tree_device.core.training_samples[sample_index];
	float3_t position												= sample.position;
	float3_t incoming_direction										= sample.incoming_direction;
	float spatial_radiance_weight									= sample.spatial_radiance_weight;

	unsigned int node_index = sample.cached_guiding_node_index;
	if (node_index == IlluminationAwareKDTreeCoreDevice::UNRESOLVED_TRAINING_SAMPLE_GUIDING_NODE_INDEX)
	{
		node_index						 = kd_tree_device.core.find_guiding_cell(position);
		sample.cached_guiding_node_index = node_index;
	}
	else if (node_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX &&
			 !(kd_tree_device.core.nodes[node_index].flags & IlluminationAwareKDTreeNodeFlag_Guiding))
	{
		// A promotion can replace the cached guiding cell between split iterations in the same post-sample update.
		node_index						 = kd_tree_device.core.find_guiding_cell(position);
		sample.cached_guiding_node_index = node_index;
	}

	if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return;

	for (unsigned int level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadLevelCount; level++)
	{
		const IlluminationAwareKDTreeNode& node = kd_tree_device.core.nodes[node_index];
		if (node.creation_tag == creation_tag)
		{
			kd_tree_device.core.atomic_add_illumination_signature(kd_tree_device.core.batch_signatures, node_index, spatial_radiance_weight,
																  incoming_direction);

			if (spatial_radiance_weight > 0.0f)
				kd_tree_device.core.atomic_add_spatial_moments(kd_tree_device.core.batch_spatial_moments, node_index, position);
		}

		if (level == IlluminationAwareKDTreeMaximumLookaheadLevelCount || !(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
			break;

		unsigned int left_child_index = node.left_child_index;

		float position_component = node.split_axis == 0 ? position.x : (node.split_axis == 1 ? position.y : position.z);
		node_index				 = position_component < node.split_position ? left_child_index : left_child_index + 1u;
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_TRAINING_SAMPLES_KERNEL_H
