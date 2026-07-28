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
inline IlluminationAwareKDTree_ReplayTrainingSamplesKernel(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int creation_tag, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ReplayTrainingSamplesKernel(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, unsigned int creation_tag)
#endif
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int sample_index = x;
#endif

	unsigned int sample_count = *illumination_aware_kd_tree.training_sample_count;
	if (sample_index >= sample_count)
		return;

	const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample = illumination_aware_kd_tree.training_samples[sample_index];

	unsigned int node_index = illumination_aware_kd_tree.find_guiding_cell(sample.position);
	for (unsigned int level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadLevelCount; level++)
	{
		const IlluminationAwareKDTreeNode& node = illumination_aware_kd_tree.nodes[node_index];
		if (node.creation_tag == creation_tag)
		{
			illumination_aware_kd_tree.atomic_add_illumination_signature(illumination_aware_kd_tree.batch_signatures, node_index, sample);

			if (sample.radiance_weight > 0.0f)
				illumination_aware_kd_tree.atomic_add_spatial_moments(illumination_aware_kd_tree.batch_spatial_moments, node_index, sample.position);
		}

		if (level == IlluminationAwareKDTreeMaximumLookaheadLevelCount || !(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
			break;

		const float* position_components = &sample.position.x;
		unsigned int left_child_index	 = node.left_child_index;

		node_index = position_components[node.split_axis] < node.split_position ? left_child_index : left_child_index + 1u;
	}
}

#endif
