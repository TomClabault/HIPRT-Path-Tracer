/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_NORMAL_FACE_OBSERVATIONS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_NORMAL_FACE_OBSERVATIONS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_AccumulateNormalFaceObservations(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_AccumulateNormalFaceObservations(IlluminationAwareKDTreeDevice kd_tree)
#endif
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int sample_index = static_cast<unsigned int>(x);
#endif

	unsigned int sample_count = *kd_tree.learning_to_cluster_training_sample_count;
	if (sample_index >= sample_count)
		return;

	const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster_training_samples[sample_index];
	if (!sample.valid_for_light_clustering)
		return;

	unsigned int guiding_node_index = kd_tree.find_guiding_cell(sample.position);
	if (guiding_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return;

	unsigned int set_index = kd_tree.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	unsigned int normal_face		= illumination_aware_kd_tree_classify_surface_normal_face(sample.shading_context.shading_normal);
	unsigned int observation_offset = kd_tree.learning_to_cluster.get_normal_face_observation_offset(set_index, normal_face);
	hippt::atomic_fetch_add(kd_tree.learning_to_cluster.normal_face_observation_counts + observation_offset, 1u);
}

#endif
