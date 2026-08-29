/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

struct IlluminationAwareKDTreeLearningToClusterTrainingSample
{
	float3_t position = float3_t(0.0f, 0.0f, 0.0f);

	IlluminationAwareKDTreeSGShadingContext shading_context{};

	unsigned int selected_cluster_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	// Stable identity of the sampled light used to determine which child owns this sample after a light-tree split.
	int emissive_triangle_global_index = -1;

	float cluster_probability  = 0.0f;
	float q_reward			   = 0.0f;
	float variance_observation = 0.0f;

	unsigned int sampled_light_clustering_index = IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX;
	unsigned int selected_cluster_slot			= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int sampled_cut_size				= 0;

	unsigned int valid_for_light_clustering = false;
};

HIPRT_DEVICE HIPRT_INLINE void IlluminationAwareKDTreeLearningToClusterDevice::append_learning_to_cluster_training_sample(
	const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample)
{
#if DirectLightSamplingStrategy != LSS_BASE_LIGHT_TREE_SG || DirectLightNEEEstimator != LSS_LEARNING_TO_CLUSTER
	return;
#endif

	unsigned int sample_index = hippt::atomic_fetch_add(training_sample_count, 0u);
	if (sample_index >= training_sample_capacity)
		return;

	sample_index = hippt::atomic_fetch_add(training_sample_count, 1u);
	if (sample_index >= training_sample_capacity)
		return;

	training_samples[sample_index]								  = sample;
	training_samples_soa.positions[sample_index]				  = sample.position;
	training_samples_soa.shading_normals[sample_index]			  = sample.shading_context.shading_normal;
	training_samples_soa.valid_for_light_clustering[sample_index] = sample.valid_for_light_clustering;
}

#endif
