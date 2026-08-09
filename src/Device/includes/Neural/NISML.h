/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_NIS_ML_H
#define DEVICE_INCLUDES_NEURAL_NIS_ML_H

#include "Device/includes/FixIntellisense.h"

#include <cstdint>
#include <math.h>

HIPRT_DEVICE uint32_t sample_nis_cluster_light_tree_sg(
	const float* cluster_log_baseline_weights, const float* residuals, uint32_t cluster_count, float random_value, float& out_probability)
{
	if (cluster_count == 0u)
	{
		out_probability = 0.0f;

		return 0u;
	}

	double maximum_combined_logit = static_cast<double>(cluster_log_baseline_weights[0]) + static_cast<double>(residuals[0]);
	for (uint32_t cluster_index = 1u; cluster_index < cluster_count; cluster_index++)
	{
		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		if (combined_logit > maximum_combined_logit)
			maximum_combined_logit = combined_logit;
	}

	float exponential_denominator = 0.0f;
	for (uint32_t cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		exponential_denominator += expf(static_cast<float>(combined_logit - maximum_combined_logit));
	}

	uint32_t final_cluster_index = cluster_count - 1u;
	double final_combined_logit	 = static_cast<double>(cluster_log_baseline_weights[final_cluster_index]) + static_cast<double>(residuals[final_cluster_index]);
	float final_probability		 = expf(static_cast<float>(final_combined_logit - maximum_combined_logit)) / exponential_denominator;
	float random_draw			 = random_value * exponential_denominator;
	float cumulative_weight		 = 0.0f;

	for (uint32_t cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		float cluster_weight  = expf(static_cast<float>(combined_logit - maximum_combined_logit));
		cumulative_weight += cluster_weight;

		if (random_draw < cumulative_weight)
		{
			out_probability = cluster_weight / exponential_denominator;

			return cluster_index;
		}
	}

	out_probability = final_probability;

	return final_cluster_index;
}

#endif
