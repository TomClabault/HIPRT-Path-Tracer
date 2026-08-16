/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_LIGHT_CLUSTERING_TRAINING_SAMPLES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
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

	unsigned int guiding_node_index = kd_tree.core.find_guiding_cell(sample.position);
	if (guiding_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return;

	unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(sample.shading_context.shading_normal);
	unsigned int set_index	 = kd_tree.core.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	unsigned int clustering_index = kd_tree.learning_to_cluster.normal_clustering_sets[set_index].clustering_indices[normal_face];
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];

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

	unsigned int iteration_budget = get_light_cluster_iteration_budget(cluster_data, kd_tree.learning_to_cluster.user_settings);
	if (iteration_budget == 0u)
		return;

	unsigned int stream_index = hippt::atomic_fetch_add(kd_tree.learning_to_cluster.reservoir_seen_counts + clustering_index, 1u);
	unsigned int target_slot  = 0u;
	bool proposes_replacement = false;

	if (stream_index < iteration_budget)
	{
		target_slot			 = stream_index;
		proposes_replacement = true;
	}
	else
	{
		unsigned int random_value = pcg_hash(sample_index ^ pcg_hash(clustering_index) ^ pcg_hash(cluster_data.iteration));
		unsigned int random_slot  = uniform_random_index(random_value, stream_index + 1u);
		if (random_slot < iteration_budget)
		{
			target_slot			 = random_slot;
			proposes_replacement = true;
		}
	}

	if (proposes_replacement)
	{
		unsigned int proposal_offset = clustering_index * kd_tree.learning_to_cluster.pending_record_stride + target_slot;
		unsigned long long int proposal =
			((static_cast<unsigned long long int>(stream_index) + 1ull) << 32) | static_cast<unsigned long long int>(sample_index);
		hippt::atomic_max(kd_tree.learning_to_cluster.reservoir_proposals + proposal_offset, proposal);
	}

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
