/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_COMMON_KERNELS_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_COMMON_KERNELS_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

HIPRT_DEVICE unsigned int compute_refinement_sampling_budget(const IlluminationAwareKDTreeLightClusteringData& cluster_data,
															 const IlluminationAwareKDTreeLearningToClusterUserSettings& settings)
{
	float growth	 = static_cast<float>(cluster_data.cut_size) / static_cast<float>(settings.initial_light_cut_size);
	float multiplier = hippt::max(growth, 2.0f);

	return static_cast<unsigned int>(ceil(multiplier * static_cast<float>(settings.initial_sampling_budget_n0)));
}

#endif
