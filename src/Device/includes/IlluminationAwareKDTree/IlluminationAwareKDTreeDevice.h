/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeCoreDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNISMLDevice.h"

struct IlluminationAwareKDTreeDevice
{
	HIPRT_DEVICE void append_learning_to_cluster_training_sample(const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample)
	{
#if DirectLightSamplingStrategy != LSS_BASE_LIGHT_TREE_SG || DirectLightNEEEstimator != LSS_SG_TREE_LEARNING_TO_CLUSTER
		return;
#endif

		unsigned int sample_index = hippt::atomic_fetch_add(learning_to_cluster_training_sample_count, 0u);
		if (sample_index >= learning_to_cluster_training_sample_capacity)
			return;

		sample_index = hippt::atomic_fetch_add(learning_to_cluster_training_sample_count, 1u);
		if (sample_index >= learning_to_cluster_training_sample_capacity)
			return;

		learning_to_cluster_training_samples[sample_index] = sample;
	}

	IlluminationAwareKDTreeCoreDevice core;
	IlluminationAwareKDTreeLearningToClusterDevice learning_to_cluster;
	IlluminationAwareKDTreeNISMLDevice nisml;

	IlluminationAwareKDTreeLearningToClusterTrainingSample* learning_to_cluster_training_samples = nullptr;
	AtomicType<unsigned int>* learning_to_cluster_training_sample_count							 = nullptr;
	unsigned int learning_to_cluster_training_sample_capacity									 = 0;
};

#endif
