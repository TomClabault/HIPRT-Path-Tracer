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
inline IlluminationAwareKDTree_LearningToClusterAccumulateNormalFaceObservations(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterAccumulateNormalFaceObservations(IlluminationAwareKDTreeDevice kd_tree)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int sample_index = static_cast<unsigned int>(x);
#endif

	unsigned int sample_count = *kd_tree.learning_to_cluster.training_sample_count;
	if (sample_index >= sample_count)
		return;

	if (kd_tree.learning_to_cluster.training_samples_soa.valid_for_lightcut[sample_index] == 0u)
		return;

	IlluminationAwareKDTreeSGShadingContext context{};
	context.position	   = kd_tree.learning_to_cluster.training_samples_soa.positions[sample_index];
	context.shading_normal = kd_tree.learning_to_cluster.training_samples_soa.shading_normals[sample_index];

	unsigned int normal_face;
	unsigned int set_index;
	kd_tree.resolve_lightcut(context, kd_tree.learning_to_cluster.training_samples_soa.mesh_ids[sample_index], nullptr, &normal_face, &set_index);
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		return;

	kd_tree.learning_to_cluster.claim_per_mesh_id_lightcut(set_index, normal_face, kd_tree.learning_to_cluster.training_samples_soa.mesh_ids[sample_index]);
	unsigned int observation_offset = kd_tree.learning_to_cluster.get_normal_face_observation_offset(set_index, normal_face);

	hippt::atomic_fetch_add(kd_tree.learning_to_cluster.normal_face_observation_counts + observation_offset, 1u);
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_NORMAL_FACE_OBSERVATIONS_H
