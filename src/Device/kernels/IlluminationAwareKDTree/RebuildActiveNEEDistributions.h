/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REBUILD_ACTIVE_NEE_DISTRIBUTIONS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_REBUILD_ACTIVE_NEE_DISTRIBUTIONS_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_RebuildActiveNEEDistributions(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
															 unsigned int tree_cut_size,
															 unsigned int active_guiding_count,
															 int x)
#else
GLOBAL_KERNEL_SIGNATURE(void)
IlluminationAwareKDTree_RebuildActiveNEEDistributions(IlluminationAwareKDTreeDevice illumination_aware_kd_tree,
													  unsigned int tree_cut_size,
													  unsigned int active_guiding_count)
#endif
{
	IlluminationAwareKDTreeNEELearntDistributionsDevice& nee_learnt_distributions = illumination_aware_kd_tree.nee_distributions;

#ifndef __KERNELCC__
	unsigned int guiding_list_index = x;
	unsigned int normal_face		= guiding_list_index % SurfaceNormalFace_Count;
	guiding_list_index /= SurfaceNormalFace_Count;
	if (guiding_list_index >= active_guiding_count)
		return;

	unsigned int guiding_node_index = illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index];
	if (guiding_node_index >= illumination_aware_kd_tree.core.node_capacity)
		return;

	unsigned int guiding_distribution_index = illumination_aware_kd_tree.core.nodes[guiding_node_index].guiding_distribution_index;
	if (guiding_distribution_index == IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX ||
		guiding_distribution_index >= illumination_aware_kd_tree.core.node_capacity)
		return;

	unsigned int distribution_index = nee_learnt_distributions.get_normal_face_distribution_index(guiding_distribution_index, normal_face);

	float learned_weights[1024] = {};
	float learned_weight_sum	= 0.0f;
	for (unsigned int slot = 0; slot < tree_cut_size; slot++)
	{
		unsigned int distribution_slot = nee_learnt_distributions.get_tree_cut_offset(distribution_index, tree_cut_size) + slot;
		float estimate				   = hippt::atomic_load(&nee_learnt_distributions.history_per_cut_node_estimated_second_moment[distribution_slot]);
		float effective_count		   = static_cast<float>(hippt::atomic_load(&nee_learnt_distributions.history_per_cut_node_sample_count[distribution_slot]));
		float batch_sum				   = hippt::atomic_load(&nee_learnt_distributions.batch_per_cut_node_second_moment_sum[distribution_slot]);
		unsigned int batch_count	   = hippt::atomic_load(&nee_learnt_distributions.batch_per_cut_node_sample_count[distribution_slot]);

		nee_learnt_distributions.update_cell_cut_node_second_moment_estimate(estimate, effective_count, batch_sum, batch_count, distribution_slot);

		learned_weights[slot] = sqrt(hippt::max(estimate, 0.0f));
		learned_weight_sum += learned_weights[slot];
	}

	float prior_mix	  = nee_learnt_distributions.compute_global_prior_mix(distribution_index);
	float running_cdf = 0.0f;
	for (unsigned int slot = 0; slot < tree_cut_size; slot++)
	{
		unsigned int distribution_slot = nee_learnt_distributions.get_tree_cut_offset(distribution_index, tree_cut_size) + slot;
		float prior_probability		   = static_cast<float>(nee_learnt_distributions.tree_cut_sampling_prior_pdfs[slot]) /
								  IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE;
		float learned_probability = prior_probability;
		if (learned_weight_sum > 0.0f)
			learned_probability = learned_weights[slot] / learned_weight_sum;

		float final_probability														= (1.0f - prior_mix) * learned_probability + prior_mix * prior_probability;
		nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot] = static_cast<unsigned short int>(
			hippt::clamp(0.0f, 1.0f, final_probability) * IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE);
		nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot] =
			static_cast<unsigned short int>(hippt::clamp(0.0f, 1.0f, running_cdf) * IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE);
		running_cdf += final_probability;
	}
#else
	unsigned int guiding_list_face_index = blockIdx.x;
	if (guiding_list_face_index >= active_guiding_count)
		return;

	unsigned int guiding_list_index = guiding_list_face_index / SurfaceNormalFace_Count;
	unsigned int normal_face		= guiding_list_face_index % SurfaceNormalFace_Count;
	unsigned int slot				= threadIdx.x;

	unsigned int guiding_node_index			= illumination_aware_kd_tree.core.active_guiding_nodes[guiding_list_index];
	unsigned int guiding_distribution_index = IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX;
	if (guiding_node_index < illumination_aware_kd_tree.core.node_capacity)
		guiding_distribution_index = illumination_aware_kd_tree.core.nodes[guiding_node_index].guiding_distribution_index;

	bool valid_distribution = guiding_distribution_index != IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX &&
							  guiding_distribution_index < illumination_aware_kd_tree.core.node_capacity;
	unsigned int distribution_index = IlluminationAwareKDTreeNode::INVALID_GUIDING_DISTRIBUTION_INDEX;
	if (valid_distribution)
		distribution_index = nee_learnt_distributions.get_normal_face_distribution_index(guiding_distribution_index, normal_face);
	bool valid_slot				   = slot < tree_cut_size;
	unsigned int distribution_slot = 0;
	if (valid_distribution && valid_slot)
		distribution_slot = nee_learnt_distributions.get_tree_cut_offset(distribution_index, tree_cut_size) + slot;

	float learned_weight = 0.0f;
	if (valid_distribution && valid_slot)
	{
		float estimate			 = hippt::atomic_load(&nee_learnt_distributions.history_per_cut_node_estimated_second_moment[distribution_slot]);
		float effective_count	 = static_cast<float>(hippt::atomic_load(&nee_learnt_distributions.history_per_cut_node_sample_count[distribution_slot]));
		float batch_estimate_sum = hippt::atomic_load(&nee_learnt_distributions.batch_per_cut_node_second_moment_sum[distribution_slot]);
		unsigned int batch_count = hippt::atomic_load(&nee_learnt_distributions.batch_per_cut_node_sample_count[distribution_slot]);

		nee_learnt_distributions.update_cell_cut_node_second_moment_estimate(estimate, effective_count, batch_estimate_sum, batch_count, distribution_slot);

		learned_weight = sqrt(hippt::max(estimate, 0.0f));
	}

	float learned_weight_sum = block_reduce<1024>(learned_weight);

	float shared_prior_mix = 0.0f;
	if (valid_distribution)
		shared_prior_mix = nee_learnt_distributions.compute_global_prior_mix(distribution_index);

	float final_probability = 0.0f;
	if (valid_distribution && valid_slot)
	{
		float prior_probability = static_cast<float>(nee_learnt_distributions.tree_cut_sampling_prior_pdfs[slot]) /
								  IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE;
		float learned_probability = prior_probability;
		if (learned_weight_sum > 0.0f)
			learned_probability = learned_weight / learned_weight_sum;

		final_probability = (1.0f - shared_prior_mix) * learned_probability + shared_prior_mix * prior_probability;
		nee_learnt_distributions.tree_cut_sampling_probabilities[distribution_slot] = static_cast<unsigned short int>(
			hippt::clamp(0.0f, 1.0f, final_probability) * IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE);
	}

	float exclusive_cdf = block_prefix_scan_exclusive<1024>(final_probability);
	if (valid_distribution && valid_slot)
		nee_learnt_distributions.tree_cut_sampling_cdfs[distribution_slot] =
			static_cast<unsigned short int>(hippt::clamp(0.0f, 1.0f, exclusive_cdf) * IlluminationAwareKDTreeNEELearntDistributionsDevice::U16_MAXIMUM_VALUE);
#endif
}

#endif
