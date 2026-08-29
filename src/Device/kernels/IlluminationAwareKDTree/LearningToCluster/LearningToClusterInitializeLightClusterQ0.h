/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTER_Q0_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTER_Q0_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/LearningToClusterCommon.h"
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
											  unsigned int lightcut_index,
											  unsigned int slot)
{
	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	if (lightcut_data.Q0_initialized || slot >= lightcut_data.lightcut_size)
		return;

	unsigned int offset									   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
	unsigned int cluster_node_index						   = kd_tree.learning_to_cluster.lightcut_node_indices[offset];
	const IlluminationAwareKDTreeSGShadingContext& context = kd_tree.learning_to_cluster.lightcut_representative_shading_contexts[lightcut_index];

	IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.lightcut_statistics[offset];
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
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int lightcut_index = blockIdx.x;
	unsigned int slot			= threadIdx.x;
	unsigned int lightcut_count = *kd_tree.learning_to_cluster.lightcut_count;
	if (lightcut_index >= lightcut_count || lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;
#else // #ifdef __KERNELCC__
	unsigned int lightcut_index = static_cast<unsigned int>(x);
	unsigned int slot			= 0u;
	if (lightcut_index >= kd_tree.learning_to_cluster.lightcut_capacity)
		return;
#endif // #ifdef __KERNELCC__

	IlluminationAwareKDTreeLightClusteringData& lightcut_data = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	unsigned int context_state								  = kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[lightcut_index];
	bool should_initialize_Q0 =
		!lightcut_data.Q0_initialized && context_state == IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY;

#ifdef __KERNELCC__
	if (should_initialize_Q0)
		initialize_light_cluster_Q0(kd_tree, light_tree_sg, lightcut_index, slot);

	if (should_initialize_Q0 && threadIdx.x == 0u)
	{
		lightcut_data.Q0_initialized	 = true;
		lightcut_data.lightcut_cdf_dirty = true;
	}
#else // #ifdef __KERNELCC__
	if (should_initialize_Q0)
	{
		for (unsigned int lightcut_slot = 0; lightcut_slot < lightcut_data.lightcut_size; lightcut_slot++)
			initialize_light_cluster_Q0(kd_tree, light_tree_sg, lightcut_index, lightcut_slot);

		lightcut_data.Q0_initialized	 = true;
		lightcut_data.lightcut_cdf_dirty = true;
	}
#endif // #ifdef __KERNELCC__
}

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_INITIALIZE_LIGHT_CLUSTER_Q0_H
