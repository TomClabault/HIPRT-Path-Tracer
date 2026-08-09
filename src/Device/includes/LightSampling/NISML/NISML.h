/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NIS_ML_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NIS_ML_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSamplingCommon.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Maths/VecTypes.h"
#include "HostDeviceCommon/Xorshift.h"

#include <math.h>

struct NISLightSample
{
	int emissive_triangle_global_index = -1;
	unsigned int cluster_index		   = 0;
	float emissive_triangle_pdf		   = 0.0f;
};

HIPRT_DEVICE LightSampleInformation sample_light_inside_nis_cluster(const HIPRTRenderData& render_data,
																	unsigned int cluster_node_index,
																	float3_t shading_point,
																	float3_t view_direction,
																	float3_t shading_normal,
																	const SGSpecularImportanceData& spec_data,
																	float specular,
																	float alpha_x,
																	float alpha_y,
																	Xorshift32Generator& random_number_generator)
{
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;
	LightSubtreeSample sampled_subtree = sample_light_tree_subtree(nodes, cluster_node_index, shading_point, view_direction, shading_normal, spec_data,
																   specular, alpha_x, alpha_y, random_number_generator);

	if (!(sampled_subtree.conditional_leaf_probability > 0.0f))
		return LightSampleInformation();

	const LightTreeSGNodeDevice& sampled_leaf = nodes[sampled_subtree.light_leaf_index];
	if (sampled_leaf.triangle_count == 0)
		return LightSampleInformation();

	int index		   = sampled_leaf.left_child_index_or_first_triangle_index + random_number_generator.random_index(sampled_leaf.triangle_count);
	int triangle_index = render_data.light_tree_sg.indices_array[index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];
	light_sample.pdf							= sampled_subtree.conditional_leaf_probability / sampled_leaf.triangle_count;

	return light_sample;
}

HIPRT_DEVICE unsigned int sample_nis_cluster(const NISMLDevice& neural_light_sampling,
											 const float* residuals,
											 Xorshift32Generator& rng,
											 float& out_cluster_probability)
{
	unsigned int cluster_count				  = neural_light_sampling.cluster_count;
	const float* cluster_log_baseline_weights = neural_light_sampling.cluster_log_baseline_weights;

	if (cluster_count == 0u)
	{
		out_cluster_probability = 0.0f;

		return 0u;
	}

	double maximum_combined_logit		  = -INFINITY;
	unsigned int last_valid_cluster_index = cluster_count;
	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		if (cluster_log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		if (combined_logit > maximum_combined_logit)
			maximum_combined_logit = combined_logit;

		last_valid_cluster_index = cluster_index;
	}
	if (last_valid_cluster_index == cluster_count)
	{
		out_cluster_probability = 0.0f;

		return cluster_count;
	}

	float exponential_denominator = 0.0f;
	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		if (cluster_log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		exponential_denominator += expf(static_cast<float>(combined_logit - maximum_combined_logit));
	}
	if (!(exponential_denominator > 0.0f))
	{
		out_cluster_probability = 0.0f;

		return cluster_count;
	}

	double final_combined_logit =
		static_cast<double>(cluster_log_baseline_weights[last_valid_cluster_index]) + static_cast<double>(residuals[last_valid_cluster_index]);
	float final_probability = expf(static_cast<float>(final_combined_logit - maximum_combined_logit)) / exponential_denominator;
	float random_value		= rng() * exponential_denominator;
	float cumulative_weight = 0.0f;

	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		if (cluster_log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		float cluster_weight  = expf(static_cast<float>(combined_logit - maximum_combined_logit));
		cumulative_weight += cluster_weight;

		if (random_value < cumulative_weight)
		{
			out_cluster_probability = cluster_weight / exponential_denominator;

			return cluster_index;
		}
	}

	out_cluster_probability = final_probability;

	return last_valid_cluster_index;
}

HIPRT_DEVICE float evaluate_nis_cluster_probability(const NISMLDevice& neural_light_sampling, const float* residuals, unsigned int target_cluster_index)
{
	unsigned int cluster_count				  = neural_light_sampling.cluster_count;
	const float* cluster_log_baseline_weights = neural_light_sampling.cluster_log_baseline_weights;

	if (target_cluster_index >= cluster_count || cluster_log_baseline_weights[target_cluster_index] == -INFINITY)
		return 0.0f;

	double maximum_combined_logit = -INFINITY;
	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		if (cluster_log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		if (combined_logit > maximum_combined_logit)
			maximum_combined_logit = combined_logit;
	}

	if (maximum_combined_logit == -INFINITY)
		return 0.0f;

	float exponential_denominator = 0.0f;
	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		if (cluster_log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		double combined_logit = static_cast<double>(cluster_log_baseline_weights[cluster_index]) + static_cast<double>(residuals[cluster_index]);
		exponential_denominator += expf(static_cast<float>(combined_logit - maximum_combined_logit));
	}

	if (!(exponential_denominator > 0.0f))
		return 0.0f;

	double target_combined_logit =
		static_cast<double>(cluster_log_baseline_weights[target_cluster_index]) + static_cast<double>(residuals[target_cluster_index]);

	return expf(static_cast<float>(target_combined_logit - maximum_combined_logit)) / exponential_denominator;
}

HIPRT_DEVICE unsigned int infer_and_sample_nis_cluster(const NISMLDevice& neural_light_sampling,
													   Xorshift32Generator& rng,
													   const float3_t& shading_point,
													   const float3_t& view_direction,
													   const float3_t& shading_normal,
													   const HIPRTRenderData& render_data,
													   float& out_cluster_probability)
{
	const float3_t& scene_min = render_data.world_settings.scene_min;
	const float3_t& scene_max = render_data.world_settings.scene_max;
	float3_t normalized_shading_point =
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
	neural_light_sampling.mlp.inference_single_thread(input, residuals);

	return sample_nis_cluster(neural_light_sampling, residuals, rng, out_cluster_probability);
}

HIPRT_DEVICE float infer_nis_cluster_probability(const NISMLDevice& neural_light_sampling,
												 unsigned int target_cluster_index,
												 const float3_t& shading_point,
												 const float3_t& view_direction,
												 const float3_t& shading_normal,
												 const HIPRTRenderData& render_data)
{
	const float3_t& scene_min = render_data.world_settings.scene_min;
	const float3_t& scene_max = render_data.world_settings.scene_max;
	float3_t normalized_shading_point =
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
	neural_light_sampling.mlp.inference_single_thread(input, residuals);

	return evaluate_nis_cluster_probability(neural_light_sampling, residuals, target_cluster_index);
}

HIPRT_DEVICE NISLightSample sample_one_emissive_triangle_neural_many_lights(const HIPRTRenderData& render_data,
																			const float3_t& shading_point,
																			const float3_t& view_direction,
																			const float3_t& shading_normal,
																			const DeviceUnpackedEffectiveMaterial& material,
																			Xorshift32Generator& random_number_generator)
{
	NISLightSample sampled_light;
	NISMLDevice neural_light_sampling	= render_data.nis_ml;
	const LightTreeSGDevice& light_tree = render_data.light_tree_sg;
	unsigned int invalid_node_index		= 0xFFFFFFFF;
	unsigned int cluster_count			= neural_light_sampling.cluster_count;

	if (cluster_count == 0 || cluster_count > NIS_MAX_CLUSTER_COUNT || light_tree.nodes == nullptr || neural_light_sampling.cluster_node_indices == nullptr)
		return sampled_light;

	float sg_specular_weight;
	float alpha_x;
	float alpha_y;
	get_sg_specular_importance_parameters(material, sg_specular_weight, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	float cluster_log_baseline_weights[NIS_MAX_CLUSTER_COUNT];
	for (unsigned int cluster_position = 0; cluster_position < cluster_count; cluster_position++)
	{
		unsigned int node_index = neural_light_sampling.cluster_node_indices[cluster_position];
		if (node_index == invalid_node_index)
		{
			cluster_log_baseline_weights[cluster_position] = -INFINITY;

			continue;
		}

		float importance = light_tree_sg_node_importance(light_tree.nodes[node_index], spec_data, shading_point, view_direction, shading_normal,
														 sg_specular_weight, alpha_x, alpha_y);
		cluster_log_baseline_weights[cluster_position] = importance > 0.0f && hippt::is_finite(importance) ? logf(importance) : -INFINITY;
	}

	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;
	float cluster_probability						   = 0.0f;
	unsigned int selected_cluster_position = infer_and_sample_nis_cluster(neural_light_sampling, random_number_generator, shading_point, view_direction,
																		  shading_normal, render_data, cluster_probability);
	if (selected_cluster_position >= cluster_count || !(cluster_probability > 0.0f))
		return sampled_light;

	unsigned int selected_cluster_node_index = neural_light_sampling.cluster_node_indices[selected_cluster_position];
	if (selected_cluster_node_index == invalid_node_index)
		return sampled_light;

	LightSampleInformation conditional_sample =
		sample_light_inside_nis_cluster(render_data, selected_cluster_node_index, shading_point, view_direction, shading_normal, spec_data, sg_specular_weight,
										alpha_x, alpha_y, random_number_generator);
	if (conditional_sample.emissive_triangle_global_index == -1 || !(conditional_sample.pdf > 0.0f))
		return sampled_light;

	sampled_light.emissive_triangle_global_index = conditional_sample.emissive_triangle_global_index;
	sampled_light.cluster_index					 = selected_cluster_position;
	sampled_light.emissive_triangle_pdf			 = cluster_probability * conditional_sample.pdf;
	if (!(sampled_light.emissive_triangle_pdf > 0.0f) || !hippt::is_finite(sampled_light.emissive_triangle_pdf))
		return NISLightSample();

	return sampled_light;
}

HIPRT_DEVICE float pdf_of_emissive_triangle_nis(const HIPRTRenderData& render_data,
												float3_t shading_point,
												float3_t view_direction,
												float3_t shading_normal,
												const DeviceUnpackedEffectiveMaterial& material,
												int global_emissive_triangle_index)
{
	NISMLDevice neural_light_sampling	= render_data.nis_ml;
	const LightTreeSGDevice& light_tree = render_data.light_tree_sg;
	unsigned int invalid_node_index		= 0xFFFFFFFF;
	unsigned int invalid_cluster_slot	= 0xFF;

	if (global_emissive_triangle_index < 0 || neural_light_sampling.cluster_count == 0 || neural_light_sampling.cluster_count > NIS_MAX_CLUSTER_COUNT ||
		light_tree.nodes == nullptr || light_tree.bit_trails == nullptr || neural_light_sampling.cluster_node_indices == nullptr ||
		neural_light_sampling.triangle_to_cluster == nullptr || neural_light_sampling.cluster_node_depths == nullptr)
		return 0.0f;

	unsigned int target_cluster_index = neural_light_sampling.triangle_to_cluster[global_emissive_triangle_index];
	if (target_cluster_index == invalid_cluster_slot || target_cluster_index >= neural_light_sampling.cluster_count)
		return 0.0f;

	unsigned int target_cluster_node_index = neural_light_sampling.cluster_node_indices[target_cluster_index];
	if (target_cluster_node_index == invalid_node_index)
		return 0.0f;

	float sg_specular_weight;
	float alpha_x;
	float alpha_y;
	get_sg_specular_importance_parameters(material, sg_specular_weight, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	float cluster_log_baseline_weights[NIS_MAX_CLUSTER_COUNT];
	for (unsigned int cluster_position = 0; cluster_position < neural_light_sampling.cluster_count; cluster_position++)
	{
		unsigned int node_index = neural_light_sampling.cluster_node_indices[cluster_position];
		if (node_index == invalid_node_index)
		{
			cluster_log_baseline_weights[cluster_position] = -INFINITY;

			continue;
		}

		float importance = light_tree_sg_node_importance(light_tree.nodes[node_index], spec_data, shading_point, view_direction, shading_normal,
														 sg_specular_weight, alpha_x, alpha_y);
		cluster_log_baseline_weights[cluster_position] = importance > 0.0f && hippt::is_finite(importance) ? logf(importance) : -INFINITY;
	}

	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;
	float cluster_probability =
		infer_nis_cluster_probability(neural_light_sampling, target_cluster_index, shading_point, view_direction, shading_normal, render_data);
	if (!(cluster_probability > 0.0f) || !hippt::is_finite(cluster_probability))
		return 0.0f;

	unsigned int bit_trail = light_tree.bit_trails[global_emissive_triangle_index];
	if (bit_trail == invalid_node_index)
		return 0.0f;

	unsigned int current_node_index		   = target_cluster_node_index;
	unsigned int current_depth			   = neural_light_sampling.cluster_node_depths[target_cluster_index];
	float conditional_triangle_probability = 1.0f;
	while (light_tree.nodes[current_node_index].triangle_count == 0)
	{
		if (current_depth >= sizeof(unsigned int) * 8)
			return 0.0f;

		unsigned int left_index	 = light_tree.nodes[current_node_index].left_child_index_or_first_triangle_index;
		unsigned int right_index = left_index + 1;
		float left_importance	 = light_tree_sg_node_importance(light_tree.nodes[left_index], spec_data, shading_point, view_direction, shading_normal,
																 sg_specular_weight, alpha_x, alpha_y);
		float right_importance	 = light_tree_sg_node_importance(light_tree.nodes[right_index], spec_data, shading_point, view_direction, shading_normal,
																 sg_specular_weight, alpha_x, alpha_y);
		float importance_sum	 = left_importance + right_importance;
		if (!(importance_sum > 0.0f))
			return 0.0f;

		float left_probability = left_importance / importance_sum;
		if ((bit_trail & (1u << current_depth)) == 0)
		{
			conditional_triangle_probability *= left_probability;
			current_node_index = left_index;
		}
		else
		{
			conditional_triangle_probability *= 1.0f - left_probability;
			current_node_index = right_index;
		}

		current_depth++;
	}

	unsigned int triangle_count = light_tree.nodes[current_node_index].triangle_count;
	if (triangle_count == 0)
		return 0.0f;

	float final_pdf = cluster_probability * conditional_triangle_probability / triangle_count;
	return final_pdf > 0.0f && hippt::is_finite(final_pdf) ? final_pdf : 0.0f;
}

#endif
