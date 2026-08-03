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
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"

HIPRT_DEVICE unsigned int compute_refinement_sampling_budget(const IlluminationAwareKDTreeLightClusteringData& cluster_data,
															 const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	float growth	 = static_cast<float>(cluster_data.cut_size) / static_cast<float>(settings.initial_light_cut_size);
	float multiplier = hippt::max(growth, 2.0f);

	return static_cast<unsigned int>(ceil(multiplier * static_cast<float>(settings.initial_sampling_budget_n0)));
}

HIPRT_DEVICE float compute_cluster_split_probability(
	float cluster_variance, float total_cut_variance, unsigned int visit_count, unsigned int cut_size, unsigned int initial_cut_size)
{
	if (!(cluster_variance > 0.0f))
		return 0.0f;

	if (visit_count <= 1u)
		return 0.0f;

	float cut_growth			   = static_cast<float>(cut_size) / static_cast<float>(initial_cut_size);
	float complexity_factor		   = 1.0f / (1.0f + cut_growth * hippt::intrin_expf(-hippt::min(cluster_variance, 80.0f)));
	float relative_variance_factor = cluster_variance / (total_cut_variance + 1.0e-6f);
	float visit_factor			   = 1.0f - 1.0f / static_cast<float>(visit_count);

	return hippt::clamp(0.0f, 1.0f, complexity_factor * relative_variance_factor * visit_factor);
}

HIPRT_DEVICE float compute_light_cluster_learning_rate(unsigned int iteration, const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	unsigned int time_step = iteration + 1u;

	return 1.0f / (settings.learning_rate_beta * hippt::intrin_pow(static_cast<float>(time_step), settings.learning_rate_omega));
}

HIPRT_DEVICE float compute_refinement_random_value(unsigned int clustering_index, unsigned int iteration, unsigned int node_index)
{
	unsigned int hash = pcg_hash(node_index);
	hash			  = pcg_hash(hash ^ iteration);
	hash			  = pcg_hash(hash ^ clustering_index);

	return static_cast<float>(hash & 0x00ffffffu) * (1.0f / 16777216.0f);
}

struct IlluminationAwareKDTreeSharedLightClusterStatistics
{
	float estimated_importance_Q;
	float estimated_second_moment;
	float variance;
	unsigned int visit_count;
};

static_assert(sizeof(IlluminationAwareKDTreeSharedLightClusterStatistics) == sizeof(IlluminationAwareKDTreeLightClusterStatistics));

// The persistent statistics type has default member initializers, which HIP does not allow for __shared__ arrays.
// Keep the shared representation trivially initialized and explicitly convert at the global-memory boundary.
HIPRT_DEVICE IlluminationAwareKDTreeLightClusterStatistics
load_light_cluster_statistics(const IlluminationAwareKDTreeSharedLightClusterStatistics& shared_statistics)
{
	IlluminationAwareKDTreeLightClusterStatistics statistics{};
	statistics.estimated_importance_Q  = shared_statistics.estimated_importance_Q;
	statistics.estimated_second_moment = shared_statistics.estimated_second_moment;
	statistics.variance				   = shared_statistics.variance;
	statistics.visit_count			   = shared_statistics.visit_count;

	return statistics;
}

HIPRT_DEVICE void store_light_cluster_statistics(IlluminationAwareKDTreeSharedLightClusterStatistics& shared_statistics,
												 const IlluminationAwareKDTreeLightClusterStatistics& statistics)
{
	shared_statistics.estimated_importance_Q  = statistics.estimated_importance_Q;
	shared_statistics.estimated_second_moment = statistics.estimated_second_moment;
	shared_statistics.variance				  = statistics.variance;
	shared_statistics.visit_count			  = statistics.visit_count;
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

	return light_tree_sg_node_importance(light_tree_sg.nodes[cluster_node_index], specular_data, context.position, context.view_direction,
										 context.shading_normal, context.sg_specular_weight, context.alpha_x, context.alpha_y);
}

HIPRT_DEVICE IlluminationAwareKDTreeLightClusterStatistics initialize_child_statistics_equation_8(const LightTreeSGDevice& light_tree_sg,
																								  unsigned int child_node_index,
																								  unsigned int sibling_node_index,
																								  const IlluminationAwareKDTreeSGShadingContext& context,
																								  const IlluminationAwareKDTreeLightClusterStatistics& parent,
																								  float learning_rate)
{
	float child_importance				= light_clustering_node_importance_for_refinement(light_tree_sg, child_node_index, context);
	float sibling_importance			= light_clustering_node_importance_for_refinement(light_tree_sg, sibling_node_index, context);
	float importance_sum				= child_importance + sibling_importance;
	float expected_child_visit_fraction = importance_sum > 0.0f ? child_importance / importance_sum : 0.5f;
	float expected_child_visit_count	= expected_child_visit_fraction * static_cast<float>(parent.visit_count);
	float history_weight				= hippt::intrin_pow(hippt::max(1.0f - learning_rate, 0.0f), expected_child_visit_count);

	IlluminationAwareKDTreeLightClusterStatistics child{};
	child.estimated_importance_Q  = history_weight * child_importance + (1.0f - history_weight) * parent.estimated_importance_Q;
	child.estimated_second_moment = child.estimated_importance_Q * child.estimated_importance_Q;
	child.variance				  = 0.0f;
	child.visit_count			  = 0;

	return child;
}

HIPRT_DEVICE bool light_clustering_refinement_is_eligible(IlluminationAwareKDTreeDevice kd_tree, unsigned int clustering_index)
{
	IlluminationAwareKDTreeLightClusteringData& cluster_data			 = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings = kd_tree.learning_to_cluster.user_settings;

	if (!settings.enable_light_cut_refinement || cluster_data.refinement_stopped || !cluster_data.Q0_initialized)
		return false;

	unsigned int context_state = *(kd_tree.learning_to_cluster.representative_shading_context_states + clustering_index);
	if (context_state != IlluminationAwareKDTreeLearningToClusterDevice::REPRESENTATIVE_SHADING_CONTEXT_STATE_READY || cluster_data.cut_size == 0u)
		return false;

	unsigned int maximum_cut_size = hippt::min(settings.maximum_light_cut_size, static_cast<unsigned int>(IlluminationAwareKDTreeMaximumLightCutSize));
	if (cluster_data.cut_size >= maximum_cut_size)
	{
		cluster_data.refinement_stopped = true;

		return false;
	}

	unsigned int sampling_budget = compute_refinement_sampling_budget(cluster_data, settings);
	if (cluster_data.refinement_sample_count < sampling_budget)
		return false;

	float inactivity_per_cluster =
		static_cast<float>(cluster_data.iteration - cluster_data.last_refinement_iteration) / static_cast<float>(cluster_data.cut_size);
	if (inactivity_per_cluster > static_cast<float>(settings.refinement_stopping_gamma))
	{
		cluster_data.refinement_stopped = true;

		return false;
	}

	return true;
}

#ifndef __KERNELCC__
HIPRT_DEVICE void refine_light_clustering_cpu(IlluminationAwareKDTreeDevice kd_tree, const LightTreeSGDevice& light_tree_sg, unsigned int guiding_list_index)
{
	unsigned int active_guiding_count = *kd_tree.active_guiding_node_count;
	if (guiding_list_index >= active_guiding_count)
		return;

	unsigned int guiding_node_index = kd_tree.active_guiding_nodes[guiding_list_index];
	unsigned int clustering_index	= kd_tree.nodes[guiding_node_index].light_clustering_index;
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	if (!light_clustering_refinement_is_eligible(kd_tree, clustering_index))
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data			  = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings  = kd_tree.learning_to_cluster.user_settings;
	const IlluminationAwareKDTreeSGShadingContext& representative_context = kd_tree.learning_to_cluster.representative_shading_contexts[clustering_index];
	unsigned int old_cut_size											  = cluster_data.cut_size;

	unsigned int old_node_indices[IlluminationAwareKDTreeMaximumLightCutSize];
	IlluminationAwareKDTreeLightClusterStatistics old_statistics[IlluminationAwareKDTreeMaximumLightCutSize];
	unsigned int new_node_indices[IlluminationAwareKDTreeMaximumLightCutSize];
	IlluminationAwareKDTreeLightClusterStatistics new_statistics[IlluminationAwareKDTreeMaximumLightCutSize];
	unsigned char split_flags[IlluminationAwareKDTreeMaximumLightCutSize];

	float total_variance = 0.0f;
	for (unsigned int slot = 0; slot < old_cut_size; slot++)
	{
		unsigned int offset	   = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		old_node_indices[slot] = kd_tree.learning_to_cluster.light_cluster_node_indices[offset];
		old_statistics[slot]   = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		total_variance += old_statistics[slot].variance;
	}

	unsigned int maximum_cut_size	  = hippt::min(settings.maximum_light_cut_size, static_cast<unsigned int>(IlluminationAwareKDTreeMaximumLightCutSize));
	unsigned int remaining_capacity	  = maximum_cut_size - old_cut_size;
	unsigned int accepted_split_count = 0;
	for (unsigned int slot = 0; slot < old_cut_size; slot++)
	{
		const LightTreeSGNodeDevice& node = light_tree_sg.nodes[old_node_indices[slot]];
		bool can_split					  = node.triangle_count == 0 && old_statistics[slot].visit_count > 1u;
		float split_probability			  = 0.0f;
		if (can_split)
			split_probability = compute_cluster_split_probability(old_statistics[slot].variance, total_variance, old_statistics[slot].visit_count, old_cut_size,
																  settings.initial_light_cut_size);

		float random_value = compute_refinement_random_value(clustering_index, cluster_data.iteration, old_node_indices[slot]);
		split_flags[slot]  = can_split && random_value < split_probability ? 1 : 0;
	}

	for (unsigned int slot = 0; slot < old_cut_size; slot++)
	{
		unsigned int proposed_splits_before = 0;
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
			float learning_rate						 = compute_light_cluster_learning_rate(cluster_data.iteration, settings);
			new_node_indices[output_slot]			 = left_child_index;
			new_node_indices[output_slot + 1u]		 = right_child_index;
			new_statistics[output_slot] = initialize_child_statistics_equation_8(light_tree_sg, left_child_index, right_child_index, representative_context,
																				 old_statistics[slot], learning_rate);
			new_statistics[output_slot + 1u] = initialize_child_statistics_equation_8(light_tree_sg, right_child_index, left_child_index,
																					  representative_context, old_statistics[slot], learning_rate);
			accepted_split_count++;
		}
	}

	unsigned int new_cut_size = old_cut_size + accepted_split_count;
	for (unsigned int slot = 0; slot < new_cut_size; slot++)
	{
		unsigned int offset												   = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		kd_tree.learning_to_cluster.light_cluster_node_indices[offset]	   = new_node_indices[slot];
		IlluminationAwareKDTreeLightClusterStatistics& statistics		   = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		statistics.estimated_importance_Q								   = new_statistics[slot].estimated_importance_Q;
		statistics.estimated_second_moment								   = new_statistics[slot].estimated_second_moment;
		statistics.variance												   = new_statistics[slot].variance;
		statistics.visit_count											   = new_statistics[slot].visit_count;
		kd_tree.learning_to_cluster.light_cluster_batch_statistics[offset] = {};
	}

	cluster_data.cut_size = new_cut_size;
	if (accepted_split_count > 0u)
		cluster_data.last_refinement_iteration = cluster_data.iteration;
	cluster_data.refinement_sample_count = 0;
}
#endif

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_RefineLightClusterings(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg, int x)
#else
HIPRT_DEVICE void refine_light_clustering_gpu(IlluminationAwareKDTreeDevice kd_tree, const LightTreeSGDevice& light_tree_sg, unsigned int guiding_list_index)
#endif
{
#ifdef __KERNELCC__
	unsigned int slot = threadIdx.x;

	unsigned int active_guiding_count = *kd_tree.active_guiding_node_count;
	if (guiding_list_index >= active_guiding_count)
		return;

	unsigned int guiding_node_index = kd_tree.active_guiding_nodes[guiding_list_index];
	unsigned int clustering_index	= kd_tree.nodes[guiding_node_index].light_clustering_index;
	if (clustering_index == IlluminationAwareKDTreeNode::INVALID_LIGHT_CLUSTERING_INDEX)
		return;

	if (!light_clustering_refinement_is_eligible(kd_tree, clustering_index))
		return;

	IlluminationAwareKDTreeLightClusteringData& cluster_data			  = kd_tree.learning_to_cluster.light_clustering_data[clustering_index];
	const IlluminationAwareKDTreeLearningToClusterUserSettings& settings  = kd_tree.learning_to_cluster.user_settings;
	const IlluminationAwareKDTreeSGShadingContext& representative_context = kd_tree.learning_to_cluster.representative_shading_contexts[clustering_index];
	unsigned int old_cut_size											  = cluster_data.cut_size;

	__shared__ unsigned int old_node_indices[IlluminationAwareKDTreeMaximumLightCutSize];
	__shared__ IlluminationAwareKDTreeSharedLightClusterStatistics old_statistics[IlluminationAwareKDTreeMaximumLightCutSize];
	__shared__ unsigned int new_node_indices[IlluminationAwareKDTreeMaximumLightCutSize];
	__shared__ IlluminationAwareKDTreeSharedLightClusterStatistics new_statistics[IlluminationAwareKDTreeMaximumLightCutSize];
	__shared__ unsigned char split_flags[IlluminationAwareKDTreeMaximumLightCutSize];

	if (slot < old_cut_size)
	{
		unsigned int offset												= kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		const IlluminationAwareKDTreeLightClusterStatistics& statistics = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		old_node_indices[slot]											= kd_tree.learning_to_cluster.light_cluster_node_indices[offset];
		store_light_cluster_statistics(old_statistics[slot], statistics);
	}
	else
	{
		old_node_indices[slot]						 = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
		old_statistics[slot].estimated_importance_Q	 = 0.0f;
		old_statistics[slot].estimated_second_moment = 0.0f;
		old_statistics[slot].variance				 = 0.0f;
		old_statistics[slot].visit_count			 = 0u;
	}

	__syncthreads();

	IlluminationAwareKDTreeLightClusterStatistics old_cluster_statistics = load_light_cluster_statistics(old_statistics[slot]);
	float local_variance												 = slot < old_cut_size ? old_cluster_statistics.variance : 0.0f;
	float total_variance												 = block_reduce<IlluminationAwareKDTreeLightClusteringBlockSize>(local_variance);

	bool can_split = false;
	if (slot < old_cut_size)
		can_split = light_tree_sg.nodes[old_node_indices[slot]].triangle_count == 0 && old_cluster_statistics.visit_count > 1u;

	float split_probability = 0.0f;
	if (can_split)
		split_probability = compute_cluster_split_probability(old_cluster_statistics.variance, total_variance, old_cluster_statistics.visit_count, old_cut_size,
															  settings.initial_light_cut_size);

	float random_value = 1.0f;
	if (slot < old_cut_size)
		random_value = compute_refinement_random_value(clustering_index, cluster_data.iteration, old_node_indices[slot]);
	split_flags[slot] = can_split && random_value < split_probability ? 1 : 0;

	unsigned int proposed_splits_before = block_prefix_scan_exclusive<IlluminationAwareKDTreeLightClusteringBlockSize>(split_flags[slot]);
	unsigned int maximum_cut_size		= hippt::min(settings.maximum_light_cut_size, static_cast<unsigned int>(IlluminationAwareKDTreeMaximumLightCutSize));
	unsigned int remaining_capacity		= maximum_cut_size - old_cut_size;
	bool split_accepted					= slot < old_cut_size && split_flags[slot] != 0 && proposed_splits_before < remaining_capacity;
	unsigned int accepted_splits_before = hippt::min(proposed_splits_before, remaining_capacity);
	unsigned int accepted_split_count	= block_reduce<IlluminationAwareKDTreeLightClusteringBlockSize>(split_accepted ? 1u : 0u);
	unsigned int output_slot			= slot + accepted_splits_before;

	if (slot < old_cut_size)
	{
		IlluminationAwareKDTreeLightClusterStatistics output_statistics{};
		if (!split_accepted)
		{
			new_node_indices[output_slot] = old_node_indices[slot];
			store_light_cluster_statistics(new_statistics[output_slot], old_cluster_statistics);
		}
		else
		{
			const LightTreeSGNodeDevice& parent_node = light_tree_sg.nodes[old_node_indices[slot]];
			unsigned int left_child_index			 = parent_node.left_child_index_or_first_triangle_index;
			unsigned int right_child_index			 = left_child_index + 1u;
			float learning_rate						 = compute_light_cluster_learning_rate(cluster_data.iteration, settings);
			new_node_indices[output_slot]			 = left_child_index;
			new_node_indices[output_slot + 1u]		 = right_child_index;
			output_statistics = initialize_child_statistics_equation_8(light_tree_sg, left_child_index, right_child_index, representative_context,
																	   old_cluster_statistics, learning_rate);
			store_light_cluster_statistics(new_statistics[output_slot], output_statistics);
			output_statistics = initialize_child_statistics_equation_8(light_tree_sg, right_child_index, left_child_index, representative_context,
																	   old_cluster_statistics, learning_rate);
			store_light_cluster_statistics(new_statistics[output_slot + 1u], output_statistics);
		}
	}

	__syncthreads();

	unsigned int new_cut_size = old_cut_size + accepted_split_count;
	if (slot < new_cut_size)
	{
		unsigned int offset												   = kd_tree.learning_to_cluster.get_light_cluster_offset(clustering_index, slot);
		kd_tree.learning_to_cluster.light_cluster_node_indices[offset]	   = new_node_indices[slot];
		IlluminationAwareKDTreeLightClusterStatistics& statistics		   = kd_tree.learning_to_cluster.light_cluster_statistics[offset];
		statistics.estimated_importance_Q								   = new_statistics[slot].estimated_importance_Q;
		statistics.estimated_second_moment								   = new_statistics[slot].estimated_second_moment;
		statistics.variance												   = new_statistics[slot].variance;
		statistics.visit_count											   = new_statistics[slot].visit_count;
		kd_tree.learning_to_cluster.light_cluster_batch_statistics[offset] = {};
	}

	__syncthreads();

	if (slot == 0u)
	{
		cluster_data.cut_size = new_cut_size;
		if (accepted_split_count > 0u)
			cluster_data.last_refinement_iteration = cluster_data.iteration;
		cluster_data.refinement_sample_count = 0;
	}
#else
	refine_light_clustering_cpu(kd_tree, light_tree_sg, static_cast<unsigned int>(x));
#endif
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_RefineLightClusterings(IlluminationAwareKDTreeDevice kd_tree, LightTreeSGDevice light_tree_sg)
{
	refine_light_clustering_gpu(kd_tree, light_tree_sg, blockIdx.x);
}
#endif

#endif
