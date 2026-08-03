/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_AccumulateLightClusteringTrainingSamples(IlluminationAwareKDTreeDevice kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_AccumulateLightClusteringTrainingSamples(IlluminationAwareKDTreeDevice kd_tree)
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

	unsigned int clustering_index = kd_tree.nodes[guiding_node_index].light_clustering_index;
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	hippt::atomic_fetch_add(kd_tree.learning_to_cluster.light_clustering_batch_sample_counts + clustering_index, 1u);

	unsigned int selected_slot = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	// TODO we should just store the cut not slot in sample in the future instead of looping like that
	for (unsigned int slot = 0; slot < cluster_data.cut_size; slot++)
	{
		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		if (kd_tree.learning_to_cluster.light_cluster_node_indices[offset] == sample.selected_cluster_node_index)
		{
			selected_slot = slot;
			break;
		}
	}

	if (selected_slot == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return;

	unsigned int selected_offset							  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, selected_slot);
	IlluminationAwareKDTreeLightClusterBatchStatistics& batch = kd_tree.learning_to_cluster.light_cluster_batch_statistics[selected_offset];
	float contribution										  = sample.light_clustering_contribution;

	hippt::atomic_fetch_add_gpu(&batch.contribution_sum, contribution);
	hippt::atomic_fetch_add_gpu(&batch.squared_contribution_sum, contribution * contribution);
	hippt::atomic_fetch_add_gpu(&batch.selected_count, 1u);

	AtomicType<unsigned int>* context_state = kd_tree.learning_to_cluster.representative_shading_context_states + clustering_index;
	unsigned int previous_state =
		hippt::atomic_compare_exchange(context_state, IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT,
									   IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_WRITING);
	if (previous_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_NO_CONTEXT)
	{
		kd_tree.learning_to_cluster.representative_shading_contexts[clustering_index] = sample.shading_context;

		__threadfence();
		hippt::atomic_exchange(context_state, IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY);
	}
}

#endif
