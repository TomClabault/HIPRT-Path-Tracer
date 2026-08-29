/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTER_Q0_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTER_Q0_H

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

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterInitializeLightClusterQ0(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterInitializeLightClusterQ0(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
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
	unsigned int context_state								 = kd_tree.learning_to_cluster.representative_shading_context_states[clustering_index];
	bool should_initialize_Q0 =
		!cluster_data.Q0_initialized && context_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY;

#ifdef __KERNELCC__
	if (should_initialize_Q0)
		initialize_light_cluster_Q0(kd_tree, light_tree_sg, clustering_index, slot);

	if (should_initialize_Q0 && threadIdx.x == 0u)
	{
		cluster_data.Q0_initialized			 = true;
		cluster_data.light_cluster_cdf_dirty = true;
	}
#else
	if (should_initialize_Q0)
	{
		for (unsigned int cluster_slot = 0; cluster_slot < cluster_data.cut_size; cluster_slot++)
			initialize_light_cluster_Q0(kd_tree, light_tree_sg, clustering_index, cluster_slot);

		cluster_data.Q0_initialized			 = true;
		cluster_data.light_cluster_cdf_dirty = true;
	}
#endif
}

#endif
