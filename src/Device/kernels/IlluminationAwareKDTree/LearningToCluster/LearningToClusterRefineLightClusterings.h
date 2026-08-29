/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REFINE_LIGHT_CLUSTERINGS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REFINE_LIGHT_CLUSTERINGS_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/IlluminationAwareKDTree/LearningToClusterCommon.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"

HIPRT_DEVICE float compute_cluster_split_probability(
	float cluster_variance, float total_lightcut_variance, unsigned int visit_count, unsigned int lightcut_size, unsigned int initial_lightcut_size)
{
	if (!(cluster_variance > 0.0f) || visit_count <= 1u)
		return 0.0f;

	float lightcut_growth		   = static_cast<float>(lightcut_size) / static_cast<float>(initial_lightcut_size);
	float complexity_factor		   = 1.0f / (1.0f + lightcut_growth * hippt::intrin_expf(-hippt::min(cluster_variance, 80.0f)));
	float relative_variance_factor = cluster_variance / (total_lightcut_variance + 1.0e-6f);
	float visit_factor			   = 1.0f - 1.0f / static_cast<float>(visit_count);

	return hippt::clamp(0.0f, 1.0f, complexity_factor * relative_variance_factor * visit_factor);
}

HIPRT_DEVICE float compute_refinement_random_value(unsigned int lightcut_index, unsigned int iteration, unsigned int node_index)
{
	unsigned int hash = pcg_hash(node_index);
	hash			  = pcg_hash(hash ^ iteration);
	hash			  = pcg_hash(hash ^ lightcut_index);

	return static_cast<float>(hash & 0x00ffffffu) * (1.0f / 16777216.0f);
}

struct IlluminationAwareKDTreeSharedLightClusterStatistics
{
	float estimated_importance_Q;
	float mean;
	float M2;
	unsigned int visit_count;
};

static_assert(sizeof(IlluminationAwareKDTreeSharedLightClusterStatistics) == sizeof(IlluminationAwareKDTreeLightClusterStatistics));

// The persistent statistics type has default member initializers, which HIP does not allow for __shared__ arrays.
// Keep the shared representation trivially initialized and explicitly convert at the global-memory boundary.
HIPRT_DEVICE IlluminationAwareKDTreeLightClusterStatistics
load_lightcut_statistics(const IlluminationAwareKDTreeSharedLightClusterStatistics& shared_statistics)
{
	IlluminationAwareKDTreeLightClusterStatistics statistics{};
	statistics.estimated_importance_Q = shared_statistics.estimated_importance_Q;
	statistics.mean					  = shared_statistics.mean;
	statistics.M2					  = shared_statistics.M2;
	statistics.visit_count			  = shared_statistics.visit_count;

	return statistics;
}

HIPRT_DEVICE void store_lightcut_statistics(IlluminationAwareKDTreeSharedLightClusterStatistics& shared_statistics,
											const IlluminationAwareKDTreeLightClusterStatistics& statistics)
{
	shared_statistics.estimated_importance_Q = statistics.estimated_importance_Q;
	shared_statistics.mean					 = statistics.mean;
	shared_statistics.M2					 = statistics.M2;
	shared_statistics.visit_count			 = statistics.visit_count;
}

HIPRT_DEVICE float light_clustering_node_importance_for_refinement(const LightTreeSGDevice& light_tree_sg,
																   unsigned int cluster_node_index,
																   const IlluminationAwareKDTreeSGShadingContext& context)
{
#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData specular_data(context.view_direction, context.shading_normal, context.alpha_x, context.alpha_y);
#else
	SGSpecularImportanceData specular_data;
#endif

	// return light_tree_sg.nodes[cluster_node_index].get_total_power();
	return hippt::max(1.0e-3f, light_tree_sg_node_importance(light_tree_sg.nodes[cluster_node_index], specular_data, context.position, context.view_direction,
															 context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y));
}

HIPRT_DEVICE IlluminationAwareKDTreeLightClusterStatistics
initialize_child_statistics_from_parent_Q(const LightTreeSGDevice& light_tree_sg,
										  unsigned int child_node_index,
										  unsigned int sibling_node_index,
										  const IlluminationAwareKDTreeSGShadingContext& context,
										  const IlluminationAwareKDTreeLightClusterStatistics& parent)
{
	float child_importance				= light_clustering_node_importance_for_refinement(light_tree_sg, child_node_index, context);
	float sibling_importance			= light_clustering_node_importance_for_refinement(light_tree_sg, sibling_node_index, context);
	float importance_sum				= child_importance + sibling_importance;
	float expected_child_visit_fraction = importance_sum > 0.0f ? child_importance / importance_sum : 0.5f;

	IlluminationAwareKDTreeLightClusterStatistics child{};
	// Q_x(c) estimates the aggregate contribution of a cut member. Partition the learned parent estimate between the children until they receive new
	// observations.
	child.estimated_importance_Q = parent.estimated_importance_Q * expected_child_visit_fraction;
	child.mean					 = 0.0f;
	child.M2					 = 0.0f;
	child.visit_count			 = 0u;

	return child;
}

HIPRT_DEVICE bool light_clustering_refinement_is_eligible(IlluminationAwareKDTreeDevice kd_tree, unsigned int lightcut_index)
{
	IlluminationAwareKDTreeLightClusteringData& lightcut_data			 = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;

	if (!settings.enable_lightcut_refinement || lightcut_data.refinement_stopped || !lightcut_data.Q0_initialized)
		return false;

	unsigned int context_state = kd_tree.learning_to_cluster.lightcut_representative_shading_context_states[lightcut_index];
	if (context_state != IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY || lightcut_data.lightcut_size == 0u)
		return false;

	unsigned int maximum_lightcut_size = LearningToClusterMaximumLightCutSize;
	if (lightcut_data.lightcut_size >= maximum_lightcut_size)
	{
		lightcut_data.refinement_stopped = true;

		return false;
	}

	unsigned int sampling_budget	   = compute_refinement_sampling_budget(lightcut_data, settings);
	unsigned int replayed_sample_count = kd_tree.learning_to_cluster.lightcut_sample_counts[lightcut_index];
	if (replayed_sample_count < sampling_budget)
		return false;

	float inactivity_per_cluster =
		static_cast<float>(lightcut_data.iteration - lightcut_data.last_refinement_iteration) / static_cast<float>(lightcut_data.lightcut_size);
	if (inactivity_per_cluster > static_cast<float>(settings.refinement_stopping_gamma))
	{
		lightcut_data.refinement_stopped = true;

		return false;
	}

	return true;
}

#ifndef __KERNELCC__
HIPRT_DEVICE void refine_light_clustering_cpu(IlluminationAwareKDTreeDevice kd_tree,
											  const LightTreeSGDevice& light_tree_sg,
											  unsigned int active_guiding_node_face_index)
{
	unsigned int active_guiding_count = *kd_tree.core.active_guiding_node_count;
	if (active_guiding_node_face_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_guiding_node_face_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_guiding_node_face_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.core.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.core.nodes[guiding_node_index].lightcut_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		return;

	unsigned int lightcut_index = kd_tree.learning_to_cluster.normal_lightcut_sets[set_index].lightcut_indices[normal_face];
	if (lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX || !light_clustering_refinement_is_eligible(kd_tree, lightcut_index))
		return;

	IlluminationAwareKDTreeLightClusteringData& lightcut_data			 = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;
	const IlluminationAwareKDTreeSGShadingContext& representative_context =
		kd_tree.learning_to_cluster.lightcut_representative_shading_contexts[lightcut_index];
	unsigned int old_lightcut_size = lightcut_data.lightcut_size;

	unsigned int old_node_indices[LearningToClusterMaximumLightCutSize];
	IlluminationAwareKDTreeLightClusterStatistics old_statistics[LearningToClusterMaximumLightCutSize];
	unsigned int new_node_indices[LearningToClusterMaximumLightCutSize];
	IlluminationAwareKDTreeLightClusterStatistics new_statistics[LearningToClusterMaximumLightCutSize];
	unsigned char split_flags[LearningToClusterMaximumLightCutSize];

	float total_variance = 0.0f;
	for (unsigned int slot = 0; slot < old_lightcut_size; slot++)
	{
		unsigned int offset	   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		old_node_indices[slot] = kd_tree.learning_to_cluster.lightcut_node_indices[offset];
		old_statistics[slot]   = kd_tree.learning_to_cluster.lightcut_statistics[offset];
		total_variance += old_statistics[slot].get_refinement_variance();
	}

	unsigned int maximum_lightcut_size = LearningToClusterMaximumLightCutSize;
	unsigned int remaining_capacity	   = maximum_lightcut_size - old_lightcut_size;
	unsigned int accepted_split_count  = 0u;
	for (unsigned int slot = 0; slot < old_lightcut_size; slot++)
	{
		const LightTreeSGNodeDevice& node = light_tree_sg.nodes[old_node_indices[slot]];
		bool can_split					  = node.triangle_count == 0 && old_statistics[slot].visit_count > 1u;
		float split_probability			  = 0.0f;
		if (can_split)
			split_probability = compute_cluster_split_probability(old_statistics[slot].get_refinement_variance(), total_variance,
																  old_statistics[slot].visit_count, old_lightcut_size, settings.initial_lightcut_size);

		float random_value = compute_refinement_random_value(lightcut_index, lightcut_data.iteration, old_node_indices[slot]);
		split_flags[slot]  = can_split && random_value < split_probability ? 1 : 0;
	}

	for (unsigned int slot = 0; slot < old_lightcut_size; slot++)
	{
		unsigned int proposed_splits_before = 0u;
		for (unsigned int previous_slot = 0; previous_slot < slot; previous_slot++)
			proposed_splits_before += split_flags[previous_slot];

		bool split_accepted					= split_flags[slot] != 0 && proposed_splits_before < remaining_capacity;
		unsigned int accepted_splits_before = hippt::min(proposed_splits_before, remaining_capacity);
		unsigned int output_slot			= slot + accepted_splits_before;

		if (!split_accepted)
		{
			new_node_indices[output_slot] = old_node_indices[slot];
			new_statistics[output_slot]	  = old_statistics[slot];
		}
		else
		{
			const LightTreeSGNodeDevice& parent_node = light_tree_sg.nodes[old_node_indices[slot]];
			unsigned int left_child_index			 = parent_node.left_child_index_or_first_triangle_index;
			unsigned int right_child_index			 = left_child_index + 1u;
			new_node_indices[output_slot]			 = left_child_index;
			new_node_indices[output_slot + 1u]		 = right_child_index;
			new_statistics[output_slot] =
				initialize_child_statistics_from_parent_Q(light_tree_sg, left_child_index, right_child_index, representative_context, old_statistics[slot]);
			new_statistics[output_slot + 1u] =
				initialize_child_statistics_from_parent_Q(light_tree_sg, right_child_index, left_child_index, representative_context, old_statistics[slot]);
			accepted_split_count++;
		}
	}

	unsigned int new_lightcut_size = old_lightcut_size + accepted_split_count;
	for (unsigned int slot = 0; slot < new_lightcut_size; slot++)
	{
		unsigned int offset										  = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		kd_tree.learning_to_cluster.lightcut_node_indices[offset] = new_node_indices[slot];
		kd_tree.learning_to_cluster.lightcut_statistics[offset]	  = new_statistics[slot];
	}

	lightcut_data.lightcut_size = new_lightcut_size;
	if (accepted_split_count > 0u)
	{
		lightcut_data.last_refinement_iteration = lightcut_data.iteration;
		lightcut_data.lightcut_cdf_dirty		= true;
	}
}
#endif // #ifndef __KERNELCC__

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_LearningToClusterRefineLightClusterings(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
HIPRT_DEVICE void refine_light_clustering_gpu(IlluminationAwareKDTreeDevice kd_tree,
											  const LightTreeSGDevice& light_tree_sg,
											  unsigned int active_guiding_node_face_index)
#endif // #ifndef __KERNELCC__
{
#ifdef __KERNELCC__
	unsigned int slot				  = threadIdx.x;
	unsigned int active_guiding_count = *kd_tree.core.active_guiding_node_count;
	if (active_guiding_node_face_index >= active_guiding_count * SurfaceNormalFace_Count)
		return;

	unsigned int guiding_list_index = active_guiding_node_face_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= active_guiding_node_face_index % SurfaceNormalFace_Count;
	unsigned int guiding_node_index = kd_tree.core.active_guiding_nodes[guiding_list_index];
	unsigned int set_index			= kd_tree.core.nodes[guiding_node_index].lightcut_normal_set_index;
	if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX)
		return;

	unsigned int lightcut_index = kd_tree.learning_to_cluster.normal_lightcut_sets[set_index].lightcut_indices[normal_face];
	if (lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX || !light_clustering_refinement_is_eligible(kd_tree, lightcut_index))
		return;

	IlluminationAwareKDTreeLightClusteringData& lightcut_data			 = kd_tree.learning_to_cluster.lightcut_data[lightcut_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;
	const IlluminationAwareKDTreeSGShadingContext& representative_context =
		kd_tree.learning_to_cluster.lightcut_representative_shading_contexts[lightcut_index];
	unsigned int old_lightcut_size = lightcut_data.lightcut_size;

	__shared__ unsigned int old_node_indices[LearningToClusterMaximumLightCutSize];
	__shared__ IlluminationAwareKDTreeSharedLightClusterStatistics old_statistics[LearningToClusterMaximumLightCutSize];
	__shared__ unsigned int new_node_indices[LearningToClusterMaximumLightCutSize];
	__shared__ IlluminationAwareKDTreeSharedLightClusterStatistics new_statistics[LearningToClusterMaximumLightCutSize];
	__shared__ unsigned char split_flags[LearningToClusterMaximumLightCutSize];

	if (slot < old_lightcut_size)
	{
		unsigned int offset	   = kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		old_node_indices[slot] = kd_tree.learning_to_cluster.lightcut_node_indices[offset];
		store_lightcut_statistics(old_statistics[slot], kd_tree.learning_to_cluster.lightcut_statistics[offset]);
	}
	else
	{
		old_node_indices[slot]						= IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
		old_statistics[slot].estimated_importance_Q = 0.0f;
		old_statistics[slot].mean					= 0.0f;
		old_statistics[slot].M2						= 0.0f;
		old_statistics[slot].visit_count			= 0u;
	}

	__syncthreads();

	IlluminationAwareKDTreeLightClusterStatistics old_cluster_statistics = load_lightcut_statistics(old_statistics[slot]);
	float local_variance												 = slot < old_lightcut_size ? old_cluster_statistics.get_refinement_variance() : 0.0f;
	float total_variance												 = block_reduce<LearningToClusterMaximumLightCutSize>(local_variance);

	bool can_split = slot < old_lightcut_size && light_tree_sg.nodes[old_node_indices[slot]].triangle_count == 0 && old_cluster_statistics.visit_count > 1u;
	float split_probability = 0.0f;
	if (can_split)
		split_probability = compute_cluster_split_probability(old_cluster_statistics.get_refinement_variance(), total_variance,
															  old_cluster_statistics.visit_count, old_lightcut_size, settings.initial_lightcut_size);

	float random_value = 1.0f;
	if (slot < old_lightcut_size)
		random_value = compute_refinement_random_value(lightcut_index, lightcut_data.iteration, old_node_indices[slot]);
	split_flags[slot] = can_split && random_value < split_probability ? 1 : 0;

	unsigned int proposed_splits_before = block_prefix_scan_exclusive<LearningToClusterMaximumLightCutSize>(split_flags[slot]);
	unsigned int maximum_lightcut_size	= LearningToClusterMaximumLightCutSize;
	unsigned int remaining_capacity		= maximum_lightcut_size - old_lightcut_size;
	bool split_accepted					= slot < old_lightcut_size && split_flags[slot] != 0 && proposed_splits_before < remaining_capacity;
	unsigned int accepted_splits_before = hippt::min(proposed_splits_before, remaining_capacity);
	unsigned int accepted_split_count	= block_reduce<LearningToClusterMaximumLightCutSize>(split_accepted ? 1u : 0u);
	unsigned int output_slot			= slot + accepted_splits_before;

	if (slot < old_lightcut_size)
	{
		if (!split_accepted)
		{
			new_node_indices[output_slot] = old_node_indices[slot];
			store_lightcut_statistics(new_statistics[output_slot], old_cluster_statistics);
		}
		else
		{
			const LightTreeSGNodeDevice& parent_node = light_tree_sg.nodes[old_node_indices[slot]];
			unsigned int left_child_index			 = parent_node.left_child_index_or_first_triangle_index;
			unsigned int right_child_index			 = left_child_index + 1u;
			new_node_indices[output_slot]			 = left_child_index;
			new_node_indices[output_slot + 1u]		 = right_child_index;
			store_lightcut_statistics(new_statistics[output_slot], initialize_child_statistics_from_parent_Q(light_tree_sg, left_child_index, right_child_index,
																											 representative_context, old_cluster_statistics));
			store_lightcut_statistics(
				new_statistics[output_slot + 1u],
				initialize_child_statistics_from_parent_Q(light_tree_sg, right_child_index, left_child_index, representative_context, old_cluster_statistics));
		}
	}

	__syncthreads();

	unsigned int new_lightcut_size = old_lightcut_size + accepted_split_count;
	if (slot < new_lightcut_size)
	{
		unsigned int offset												= kd_tree.learning_to_cluster.get_light_cluster_offset(lightcut_index, slot);
		kd_tree.learning_to_cluster.lightcut_node_indices[offset]		= new_node_indices[slot];
		IlluminationAwareKDTreeLightClusterStatistics& statistics		= kd_tree.learning_to_cluster.lightcut_statistics[offset];
		IlluminationAwareKDTreeLightClusterStatistics output_statistics = load_lightcut_statistics(new_statistics[slot]);
		statistics.estimated_importance_Q								= output_statistics.estimated_importance_Q;
		statistics.mean													= output_statistics.mean;
		statistics.M2													= output_statistics.M2;
		statistics.visit_count											= output_statistics.visit_count;
	}

	__syncthreads();

	if (slot == 0u)
	{
		lightcut_data.lightcut_size = new_lightcut_size;
		if (accepted_split_count > 0u)
		{
			lightcut_data.last_refinement_iteration = lightcut_data.iteration;
			lightcut_data.lightcut_cdf_dirty		= true;
		}
	}
#else // #ifdef __KERNELCC__
	refine_light_clustering_cpu(kd_tree, light_tree_sg, static_cast<unsigned int>(x));
#endif // #ifdef __KERNELCC__
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_LearningToClusterRefineLightClusterings(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
{
	refine_light_clustering_gpu(kd_tree, light_tree_sg, blockIdx.x);
}
#endif // #ifdef __KERNELCC__

#endif // #ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REFINE_LIGHT_CLUSTERINGS_H
