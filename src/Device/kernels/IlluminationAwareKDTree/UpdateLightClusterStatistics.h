/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_UPDATE_LIGHT_CLUSTER_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_UPDATE_LIGHT_CLUSTER_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"

HIPRT_DEVICE float light_clustering_node_importance(const LightTreeSGDevice& light_tree_sg,
													unsigned int cluster_node_index,
													const IlluminationAwareKDTreeSGShadingContext& context)
{
#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
	SGSpecularImportanceData specular_data;
#endif

	return light_tree_sg_node_importance(light_tree_sg.nodes[cluster_node_index], specular_data, context.position, context.view_direction,
										 context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);
}

HIPRT_DEVICE void initialize_light_cluster_Q_from_Lu(IlluminationAwareKDTreeDevice kd_tree,
													 const LightTreeSGDevice& light_tree_sg,
													 unsigned int clustering_index,
													 unsigned int slot)
{
	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	if (cluster_data.Q0_initialized || slot >= cluster_data.cut_size)
		return;

	unsigned int offset									   = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
	unsigned int cluster_node_index						   = kd_tree.learning_to_cluster.light_cluster_node_indices[offset];
	const IlluminationAwareKDTreeSGShadingContext& context = kd_tree.learning_to_cluster.representative_shading_contexts[clustering_index];

	// Q0 is used as the cluster sampling distribution. Keep it tied to the current shading context instead of assigning power to clusters that cannot be
	// sampled at this point.
	float initial_importance_Q = light_clustering_node_importance(light_tree_sg, cluster_node_index, context);

	IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
	statistics.estimated_importance_Q						  = initial_importance_Q;
	statistics.estimated_second_moment						  = initial_importance_Q * initial_importance_Q;
	statistics.variance										  = 0.0f;
	statistics.visit_count									  = 0;
}

HIPRT_DEVICE void update_light_cluster_statistics_for_slot(IlluminationAwareKDTreeDevice kd_tree,
														   unsigned int clustering_index,
														   unsigned int slot,
														   unsigned int total_sample_count)
{
	IlluminationAwareKDTreeLightClusteringData& cluster_data  = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
	IlluminationAwareKDTreeLightClusterStatistics& persistent = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
	IlluminationAwareKDTreeLightClusterBatchStatistics batch  = kd_tree.learning_to_cluster.light_cluster_batch_statistics.read(offset);

	float inverse_sample_count = 1.0f / static_cast<float>(total_sample_count);
	float batch_mean		   = batch.contribution_sum * inverse_sample_count;
	float batch_second_moment  = batch.squared_contribution_sum * inverse_sample_count;
	unsigned int iteration	   = cluster_data.iteration + 1u;
	float learning_rate		   = 1.0f / (kd_tree.learning_to_cluster.user_settings.learning_rate_beta *
									 hippt::intrin_pow(static_cast<float>(iteration), kd_tree.learning_to_cluster.user_settings.learning_rate_omega));

	persistent.estimated_importance_Q  = (1.0f - learning_rate) * persistent.estimated_importance_Q + learning_rate * batch_mean;
	persistent.estimated_second_moment = (1.0f - learning_rate) * persistent.estimated_second_moment + learning_rate * batch_second_moment;
	persistent.variance = hippt::max(persistent.estimated_second_moment - persistent.estimated_importance_Q * persistent.estimated_importance_Q, 0.0f);
	persistent.visit_count += batch.selected_count;

	kd_tree.learning_to_cluster.light_cluster_batch_statistics.reset(offset);
}

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_UpdateLightClusterStatistics(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_UpdateLightClusterStatistics(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
#endif
{
#ifdef __KERNELCC__
	unsigned int active_pair_index = blockIdx.x;
	unsigned int slot			   = threadIdx.x;
#else
	unsigned int active_pair_index = static_cast<unsigned int>(x);
	unsigned int slot			   = 0;
#endif

	unsigned int active_guiding_count = *kd_tree.active_guiding_node_count;
	if (active_pair_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_pair_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_pair_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	unsigned int clustering_index = kd_tree.learning_to_cluster.normal_clustering_sets[set_index].clustering_indices[normal_face];
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	unsigned int total_sample_count							 = kd_tree.learning_to_cluster.light_clustering_batch_sample_counts[clustering_index];
	if (total_sample_count == 0)
		return;

	unsigned int context_state = kd_tree.learning_to_cluster.representative_shading_context_states[clustering_index];

#ifdef __KERNELCC__
	if (!cluster_data.Q0_initialized && context_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY)
		initialize_light_cluster_Q_from_Lu(kd_tree, light_tree_sg, clustering_index, slot);

	__syncthreads();

	if (slot == 0 && !cluster_data.Q0_initialized &&
		context_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY)
		cluster_data.Q0_initialized = true;

	__syncthreads();

	if (slot < cluster_data.cut_size)
		update_light_cluster_statistics_for_slot(kd_tree, clustering_index, slot, total_sample_count);

	__syncthreads();

	if (slot == 0)
#else
	if (!cluster_data.Q0_initialized && context_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY)
	{
		for (unsigned int cluster_slot = 0; cluster_slot < cluster_data.cut_size; cluster_slot++)
			initialize_light_cluster_Q_from_Lu(kd_tree, light_tree_sg, clustering_index, cluster_slot);

		cluster_data.Q0_initialized = true;
	}

	for (unsigned int cluster_slot = 0; cluster_slot < cluster_data.cut_size; cluster_slot++)
		update_light_cluster_statistics_for_slot(kd_tree, clustering_index, cluster_slot, total_sample_count);
#endif
	{
		cluster_data.iteration++;
		cluster_data.refinement_sample_count += total_sample_count;
		kd_tree.learning_to_cluster.light_clustering_batch_sample_counts[clustering_index] = 0;
	}
}

#endif
