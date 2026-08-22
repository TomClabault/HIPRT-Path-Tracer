/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_UPDATE_LIGHT_CLUSTER_STATISTICS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_UPDATE_LIGHT_CLUSTER_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/CommonKernels.h"
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

HIPRT_DEVICE void initialize_light_cluster_Q0(IlluminationAwareKDTreeDevice kd_tree,
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

	IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
#if LearningToClusterQ0UseTotalPower == KERNEL_OPTION_TRUE
	statistics.estimated_importance_Q = light_tree_sg.nodes[cluster_node_index].get_total_power();
#else
	statistics.estimated_importance_Q = hippt::max(1.0e-3f, light_clustering_node_importance(light_tree_sg, cluster_node_index, context));
#endif
	statistics.mean		   = 0.0f;
	statistics.M2		   = 0.0f;
	statistics.visit_count = 0u;
}

HIPRT_DEVICE void append_observation(IlluminationAwareKDTreeLightClusterStatistics& statistics, float observation)
{
	unsigned int previous_count = statistics.visit_count;

	float delta = observation - statistics.mean;
	statistics.mean += delta / static_cast<float>(previous_count + 1u);

	float delta2 = observation - statistics.mean;
	statistics.M2 += delta * delta2;
	statistics.visit_count = previous_count + 1u;
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

	if (slot != 0u)
		return;

	unsigned int active_guiding_count = *kd_tree.core.active_guiding_node_count;
	if (active_pair_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_pair_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_pair_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.core.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.core.nodes[guiding_node_index].light_clustering_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	unsigned int clustering_index = kd_tree.learning_to_cluster.normal_clustering_sets[set_index].clustering_indices[normal_face];
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];

	unsigned int context_state = kd_tree.learning_to_cluster.representative_shading_context_states[clustering_index];
	if (!cluster_data.Q0_initialized && context_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY)
	{
		for (unsigned int cluster_slot = 0; cluster_slot < cluster_data.cut_size; cluster_slot++)
			initialize_light_cluster_Q0(kd_tree, light_tree_sg, clustering_index, cluster_slot);

		cluster_data.Q0_initialized = true;
	}

	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;
	unsigned int pending_count											 = kd_tree.learning_to_cluster.pending_light_cluster_record_counts[clustering_index];
	unsigned int iteration_budget										 = get_light_cluster_iteration_budget(cluster_data, settings);
	if (pending_count < iteration_budget)
		return;

	unsigned int base_offset = clustering_index * kd_tree.learning_to_cluster.pending_record_stride;
	for (unsigned int record_index = 0; record_index < pending_count; record_index++)
	{
		const IlluminationAwareKDTreePendingLightClusterRecord& record = kd_tree.learning_to_cluster.pending_light_cluster_records[base_offset + record_index];
		int slot_index												   = find_light_cluster_slot(kd_tree, clustering_index, record.cluster_node_index);
		if (slot_index < 0)
			continue;

		unsigned int offset = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, static_cast<unsigned int>(slot_index));
		append_observation(kd_tree.learning_to_cluster.light_cluster_statistics[offset], record.variance_observation);
	}
}

#endif
