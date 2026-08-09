/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_NIS_ML_H
#define DEVICE_INCLUDES_NEURAL_NIS_ML_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"
#include "HostDeviceCommon/Xorshift.h"

#include <cstdint>
#include <math.h>

HIPRT_DEVICE uint32_t sample_nis_cluster(
	const float* residuals, const float* cluster_log_baseline_weights, uint32_t cluster_count, Xorshift32Generator& rng, float& out_cluster_probability)
{
	if (cluster_count == 0u)
	{
		out_cluster_probability = 0.0f;

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
	float random_draw			 = rng() * exponential_denominator;
	float cumulative_weight		 = 0.0f;

	for (uint32_t cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		float cluster_weight  = expf(static_cast<float>(combined_logit - maximum_combined_logit));
		cumulative_weight += cluster_weight;

		if (random_draw < cumulative_weight)
		{
			out_cluster_probability = cluster_weight / exponential_denominator;

			return cluster_index;
		}
	}

	out_cluster_probability = final_probability;

	return final_cluster_index;
}

HIPRT_DEVICE uint32_t infer_and_sample_nis_cluster(const NeuralImportanceSamplingMLP& mlp,
												   const float* cluster_log_baseline_weights,
												   uint32_t cluster_count,
												   Xorshift32Generator& rng,
												   const float3_t& shading_point,
												   const float3_t& view_direction,
												   const float3_t& shading_normal,
												   const float3_t& scene_min,
												   const float3_t& scene_max,
												   float& out_cluster_probability)
{
	const float3_t normalized_shading_point =
		make_float3((shading_point.x - scene_min.x) / (scene_max.x - scene_min.x), (shading_point.y - scene_min.y) / (scene_max.y - scene_min.y),
					(shading_point.z - scene_min.z) / (scene_max.z - scene_min.z));

	NeuralImportanceSamplingMLP::InputLayer input;
	input.input[0] = normalized_shading_point.x;
	input.input[1] = normalized_shading_point.y;
	input.input[2] = normalized_shading_point.z;
	input.input[3] = view_direction.x;
	input.input[4] = view_direction.y;
	input.input[5] = view_direction.z;
	input.input[6] = shading_normal.x;
	input.input[7] = shading_normal.y;
	input.input[8] = shading_normal.z;

	float residuals[NIS_MAX_CLUSTER_COUNT];
	mlp.inference_single_thread(input, residuals);

	return sample_nis_cluster(residuals, cluster_log_baseline_weights, cluster_count, rng, out_cluster_probability);
}

#endif
