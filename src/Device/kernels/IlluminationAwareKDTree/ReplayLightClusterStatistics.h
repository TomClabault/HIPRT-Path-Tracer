/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REPLAY_LIGHT_CLUSTER_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "Device/kernels/IlluminationAwareKDTree/ReplayLightClusterTrainingSamples.h"

HIPRT_DEVICE void append_replayed_light_cluster_observation(IlluminationAwareKDTreeLightClusterStatistics& statistics, float observation)
{
	unsigned int previous_count = statistics.visit_count;
	float delta					= observation - statistics.mean;
	statistics.mean += delta / static_cast<float>(previous_count + 1u);

	float delta2 = observation - statistics.mean;
	statistics.M2 += delta * delta2;
	statistics.visit_count = previous_count + 1u;
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_ReplayLightClusterStatistics(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_ReplayLightClusterStatistics(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
#endif
{
#ifdef __KERNELCC__
	unsigned int clustering_index		= blockIdx.x;
	unsigned int slot					= threadIdx.x;
	unsigned int light_clustering_count = *kd_tree.learning_to_cluster.light_clustering_count;
	if (clustering_index >= light_clustering_count || clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;
#else
	unsigned int clustering_index = static_cast<unsigned int>(x);
	unsigned int slot			  = 0u;
	if (clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;
#endif

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	if (!cluster_data.Q0_initialized)
		return;

	unsigned int sample_count = *kd_tree.learning_to_cluster.training_sample_count;
#ifdef __KERNELCC__
	if (threadIdx.x == 0u)
		kd_tree.learning_to_cluster.light_cluster_sample_counts[clustering_index] = 0u;
	__syncthreads();

	if (slot < cluster_data.cut_size)
	{
		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);

		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];

		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];

			unsigned int sample_clustering_index = get_replayed_light_clustering_index(kd_tree, sample);
			if (sample_clustering_index != clustering_index)
				continue;

			int sample_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, clustering_index, sample);
			if (sample_slot != static_cast<int>(slot))
				continue;

			append_replayed_light_cluster_observation(statistics, sample.variance_observation);

			hippt::atomic_fetch_add(kd_tree.learning_to_cluster.light_cluster_sample_counts + clustering_index, 1u);
		}
	}
#else
	kd_tree.learning_to_cluster.light_cluster_sample_counts[clustering_index] = 0u;
	for (slot = 0u; slot < cluster_data.cut_size; slot++)
	{
		unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster.training_samples[sample_index];
			unsigned int sample_clustering_index								 = get_replayed_light_clustering_index(kd_tree, sample);
			if (sample_clustering_index != clustering_index)
				continue;

			int sample_slot = find_replayed_light_cluster_slot(kd_tree, light_tree_sg, clustering_index, sample);
			if (sample_slot != static_cast<int>(slot))
				continue;

			append_replayed_light_cluster_observation(statistics, sample.variance_observation);
			kd_tree.learning_to_cluster.light_cluster_sample_counts[clustering_index]++;
		}
	}
#endif
}

#endif
