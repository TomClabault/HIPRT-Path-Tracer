/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_COMMIT_LIGHT_CLUSTER_RESERVOIR_PROPOSALS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

HIPRT_DEVICE void commit_light_cluster_reservoir_proposal(IlluminationAwareKDTreeDevice kd_tree, unsigned int clustering_index, unsigned int slot)
{
	unsigned int proposal_offset	= clustering_index * kd_tree.learning_to_cluster.pending_record_stride + slot;
	unsigned long long int proposal = kd_tree.learning_to_cluster.reservoir_proposals[proposal_offset];

	if (proposal == 0ull)
		return;

	unsigned int stream_index = static_cast<unsigned int>((proposal >> 32) - 1ull);
	unsigned int sample_index = static_cast<unsigned int>(proposal & 0xffffffffull);
	unsigned int sample_count = *kd_tree.learning_to_cluster_training_sample_count;

	if (sample_index >= sample_count)
		return;

	const IlluminationAwareKDTreeLearningToClusterTrainingSample& sample = kd_tree.learning_to_cluster_training_samples[sample_index];
	IlluminationAwareKDTreePendingLightClusterRecord& record			 = kd_tree.learning_to_cluster.pending_light_cluster_records[proposal_offset];
	record.cluster_node_index											 = sample.selected_cluster_node_index;
	record.q_reward														 = sample.q_reward;
	record.variance_observation											 = sample.variance_observation;
	record.stream_index													 = stream_index;
}

HIPRT_DEVICE void finalize_light_cluster_reservoir(IlluminationAwareKDTreeDevice kd_tree, unsigned int clustering_index)
{
	IlluminationAwareKDTreeLightClusteringData& cluster_data			 = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;
	unsigned int iteration_budget										 = get_light_cluster_iteration_budget(cluster_data, settings);
	unsigned int seen_count												 = kd_tree.learning_to_cluster.reservoir_seen_counts[clustering_index];

	cluster_data.pending_record_budget												  = iteration_budget;
	kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index] = hippt::min(seen_count, iteration_budget);
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_CommitLightClusterReservoirProposals(IlluminationAwareKDTreeDevice kd_tree, int x)
{
	unsigned int clustering_index = static_cast<unsigned int>(x);
	if (clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity)
		return;

	for (unsigned int slot = 0; slot < kd_tree.learning_to_cluster.pending_record_stride; slot++)
		commit_light_cluster_reservoir_proposal(kd_tree, clustering_index, slot);

	finalize_light_cluster_reservoir(kd_tree, clustering_index);
}
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_CommitLightClusterReservoirProposals(IlluminationAwareKDTreeDevice kd_tree)
{
	unsigned int clustering_index = blockIdx.x;
	unsigned int slot			  = threadIdx.x;

	if (clustering_index >= kd_tree.learning_to_cluster.light_clustering_capacity || slot >= kd_tree.learning_to_cluster.pending_record_stride)
		return;

	commit_light_cluster_reservoir_proposal(kd_tree, clustering_index, slot);
	__syncthreads();

	if (slot == 0u)
		finalize_light_cluster_reservoir(kd_tree, clustering_index);
}
#endif

#endif
