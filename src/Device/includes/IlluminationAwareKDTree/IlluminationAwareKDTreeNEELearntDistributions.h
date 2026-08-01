/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NEE_LEARN_DISTRIBUTIONS_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NEE_LEARN_DISTRIBUTIONS_H

#include "Device/includes/CDF.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDirectIlluminationTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningNEESettings.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSampledCutNode.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

struct IlluminationAwareKDTreeNEEDistributionTrainingRecord
{
	// Used after spatial splitting to locate the final guiding cell.
	float3_t shading_position;

	// Used to measure whether the cell contains compatible surface orientations.
	float3_t shading_normal;

	// Global cut slot selected for this sample.
	unsigned int selected_cut_slot;

	// One stochastic observation of NEEEstimate / pLightSampleSubtree / pLightSamplePoint, which is the contribution conditioned on having selected a given cut
	// node.
	//
	// Note that this is not simply the full NEE estimate because this would have the distributions learn from average contributions, which is not optimal for
	// variance reduction. Instead, the distributions learn from the second moment of the contribution conditioned on having selected a given cut node, which is
	// the optimal objective for variance reduction, i.e. we learn from (NEEEstimate / pLightSampleSubtree / pLightSamplePoint)^2 which doesn't contain the
	// top-level cut node selection probability.
	float conditional_second_moment_contribution;
};

struct IlluminationAwareKDTreeNEELearntDistributions
{
	static constexpr float TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE = -1.0f;
	// When initializing a cell distribution with the prior distribution, how many samples that prio-distribution-initialization is going to be worth. This is
	// basically as if the cell had learnt the prior distribution from this many samples.
	static constexpr float ROOT_PRIOR_STRENGTH = 8.0f;

	HIPRT_DEVICE unsigned int get_tree_cut_offset(unsigned int guiding_distribution_index, unsigned int tree_cut_size) const
	{
		return guiding_distribution_index * tree_cut_size;
	}

	HIPRT_DEVICE void append_nee_distribution_training_record(const IlluminationAwareKDTreeNEEDistributionTrainingRecord& record)
	{
#if DirectLightSamplingStrategy != LSS_BASE_LIGHT_TREE_SG || DirectLightNEEEstimator != LSS_SG_TREE_LEARNT_DISTRIBUTIONS
		return;
#endif

		unsigned int record_index = hippt::atomic_fetch_add(nee_training_record_count, 0u);
		// The counter may exceed capacity, but memory must never be written outside the allocated buffer.
		if (record_index >= nee_training_record_capacity)
			return;

		record_index = hippt::atomic_fetch_add(nee_training_record_count, 1u);
		if (record_index >= nee_training_record_capacity)
			return;

		nee_training_records[record_index] = record;
	}

	HIPRT_DEVICE IlluminationAwareKDTreeSampledCutNode sample_global_cut_node(const LightTreeSGDevice& light_tree_sg,
																			  unsigned int guiding_distribution_index,
																			  Xorshift32Generator& random_number_generator) const
	{
		IlluminationAwareKDTreeSampledCutNode result{};
		unsigned int tree_cut_size = light_tree_sg.settings.tree_cut_size;
		if (tree_cut_size == 0)
			return result;

		unsigned int tree_cut_offset = get_tree_cut_offset(guiding_distribution_index, tree_cut_size);
		CDFDevice tree_cut_cdf;
		tree_cut_cdf.cdf  = tree_cut_sampling_cdfs + tree_cut_offset;
		tree_cut_cdf.size = tree_cut_size;

		if (tree_cut_cdf.cdf[0] == TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE)
		{
			// The distribution has not been initialized yet, return the first slot as a fallback
			result.cut_slot				 = 0;
			result.light_tree_node_index = 0;
			result.probability			 = TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE;

			return result;
		}

		unsigned int selected_slot	 = tree_cut_cdf.sample(random_number_generator);
		selected_slot				 = hippt::min(selected_slot, tree_cut_size - 1);
		unsigned int selected_offset = tree_cut_offset + selected_slot;

		result.cut_slot				 = selected_slot;
		result.light_tree_node_index = light_tree_sg.tree_cut_node_indices[selected_slot];
		result.probability			 = tree_cut_sampling_probabilities[selected_offset];

		return result;
	}

	HIPRT_DEVICE IlluminationAwareKDTreeNEEDistributionTrainingRecord make_nee_distribution_training_record(
		const IlluminationAwareKDTreeSampledCutNode& guided_sample, float3_t shading_position, float3_t shading_normal, ColorRGB32F full_local_nee_estimator)
	{
		IlluminationAwareKDTreeNEEDistributionTrainingRecord record{};

		// The complete estimator contains division by (p_cut * q_subtree * p_point)
		//
		// Multiplying by p_cut removes that probability and what remains measures the contribution conditioned on having selected this cut node which is our
		// learning objective
		ColorRGB32F conditional_estimator = full_local_nee_estimator * guided_sample.probability;

		float contribution_squared = conditional_estimator.r * conditional_estimator.r + conditional_estimator.g * conditional_estimator.g +
									 conditional_estimator.b * conditional_estimator.b;

		record.shading_position						  = shading_position;
		record.shading_normal						  = hippt::normalize(shading_normal);
		record.selected_cut_slot					  = guided_sample.cut_slot;
		record.conditional_second_moment_contribution = contribution_squared;

		return record;
	}

	/**
	 * From the existing estimate of second moment and the number of samples that have been accumulated for that estimate (in a given guiding cell * cut node),
	 * update the estimate with a new batch of samples and the result is returned in the same variables. The effective count is clamped to a maximum value to
	 * prevent the estimate from becoming too rigid and unable to adapt
	 */
	HIPRT_DEVICE void update_cell_cut_node_second_moment_estimate(float& in_out_estimate,
																  float& in_out_effective_count,
																  float batch_sum,
																  unsigned int batch_count)
	{
		// No samples selected this cut slot during the current batch.
		if (batch_count == 0)
			return;

		float old_count		 = hippt::min(in_out_effective_count, learning_nee_settings.maximum_effective_count);
		float incoming_count = static_cast<float>(batch_count);
		float denominator	 = old_count + incoming_count;

		if (!(denominator > 0.0f))
			return;

		in_out_estimate		   = (old_count * in_out_estimate + batch_sum) / denominator;
		in_out_effective_count = hippt::min(denominator, learning_nee_settings.maximum_effective_count);
	}

	IlluminationAwareKDTreeLearningNEESettings learning_nee_settings;

	// NEE samples gathered during path tracing used for training distributions
	IlluminationAwareKDTreeNEEDistributionTrainingRecord* nee_training_records = nullptr;
	AtomicType<unsigned int>* nee_training_record_count						   = nullptr;
	unsigned int nee_training_record_capacity								   = 0;

	// SG Light tree tree cut size * node capacity in size. Should be indexed by a guiding distribution index. Gives access to a tree cut size long array of
	// probabilities for sampling the nodes of the tree cut of the SG light tree.
	float* tree_cut_sampling_probabilities = nullptr;
	float* tree_cut_sampling_cdfs		   = nullptr;

	// For each guiding cell (no cut node here), how many samples have been accumulated in that cell, accross all cut nodes
	AtomicType<unsigned int>* history_per_cell_sample_count = nullptr;
	// For each guiding cell, the sum of all observed shading normals and how many normals have been observed.
	AtomicType<float>* history_per_cell_normal_sum_x		= nullptr;
	AtomicType<float>* history_per_cell_normal_sum_y		= nullptr;
	AtomicType<float>* history_per_cell_normal_sum_z		= nullptr;
	AtomicType<unsigned int>* history_per_cell_normal_count = nullptr;
	// For each cut node * guiding cell, the history of all observed second moments of the NEE estimator + how many samples have been observed.
	AtomicType<float>* history_per_cut_node_estimated_second_moment = nullptr;
	AtomicType<unsigned int>* history_per_cut_node_sample_count		= nullptr;
	// For each cut node * guiding cell, the sum of all observed second moments of the NEE estimator and how many samples have been observed in the current
	// batch (current SPP)
	AtomicType<float>* batch_per_cut_node_second_moment_sum	  = nullptr;
	AtomicType<unsigned int>* batch_per_cut_node_sample_count = nullptr;

	// Global prior distribution for sampling the tree cut nodes.
	float* tree_cut_sampling_prior_pdfs = nullptr;
	float* tree_cut_sampling_prior_cdfs = nullptr;
};

#endif
