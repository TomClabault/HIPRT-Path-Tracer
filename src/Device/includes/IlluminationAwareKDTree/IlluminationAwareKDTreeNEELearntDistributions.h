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

struct IlluminationAwareKDTreeNEELearnDistributions
{
	static constexpr float TREE_CUT_SAMPLING_DISTRIBUTION_UNINITIALIZED_VALUE = -1.0f;
	static constexpr float ROOT_PRIOR_STRENGTH								  = 8.0f;

	IlluminationAwareKDTreeLearningNEESettings learning_nee_settings;

	// SG Light tree tree cut size * node capacity in size. Should be indexed by a guiding distribution index. Gives access to a tree cut size long array of
	// probabilities for sampling the nodes of the tree cut of the SG light tree.
	float* tree_cut_sampling_probabilities = nullptr;
	float* tree_cut_sampling_cdfs		   = nullptr;
	float* estimated_second_moment		   = nullptr;
	float* effective_sample_count		   = nullptr;
	float* batch_second_moment_sum		   = nullptr;
	unsigned int* batch_sample_count	   = nullptr;

	float* tree_cut_sampling_prior_pdfs = nullptr;
	float* tree_cut_sampling_prior_cdfs = nullptr;

	HIPRT_DEVICE unsigned int get_tree_cut_offset(unsigned int guiding_distribution_index, unsigned int tree_cut_size) const
	{
		return guiding_distribution_index * tree_cut_size;
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
};

#endif
