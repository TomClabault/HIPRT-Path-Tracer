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
#else // #ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CoreReplayTrainingSamples(IlluminationAwareKDTreeDevice kd_tree_device, unsigned int creation_tag)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else // #ifdef __KERNELCC__
	unsigned int sample_index = x;
#endif // #ifdef __KERNELCC__

	unsigned int sample_count = *kd_tree_device.core.training_sample_count;
	if (sample_index >= sample_count)
		return;

	const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample = kd_tree_device.core.training_samples[sample_index];

	unsigned int node_index = kd_tree_device.core.find_guiding_cell(sample.position);
	for (unsigned int level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadLevelCount; level++)
	{
		const IlluminationAwareKDTreeNode& node = kd_tree_device.core.nodes[node_index];
		if (node.creation_tag == creation_tag)
		{
			kd_tree_device.core.atomic_add_illumination_signature(kd_tree_device.core.batch_signatures, node_index, sample);

			if (sample.spatial_radiance_weight > 0.0f)
				kd_tree_device.core.atomic_add_spatial_moments(kd_tree_device.core.batch_spatial_moments, node_index, sample.position);
		}

		if (level == IlluminationAwareKDTreeMaximumLookaheadLevelCount || !(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
			break;

		const float* position_components = &sample.position.x;
		unsigned int left_child_index	 = node.left_child_index;

		node_index = position_components[node.split_axis] < node.split_position ? left_child_index : left_child_index + 1u;
	}
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_TRAINING_SAMPLES_KERNEL_H
