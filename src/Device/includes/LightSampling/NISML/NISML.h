/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NIS_ML_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NIS_ML_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSamplingCommon.h"
#include "Device/includes/Neural/NISML/NISMLPositionLearnableDenseGrid.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

#include <math.h>

struct NISLightSample
{
	int emissive_triangle_global_index = -1;

	unsigned int cluster_index = 0;

	float cluster_probability			= 0.0f;
	float conditional_light_probability = 0.0f;
	float emissive_triangle_pdf			= 0.0f;
};

HIPRT_DEVICE void build_nis_input(const NISMLPositionLearnableDenseGridDevice& position_learnable_dense_grid,
								  const float3_t& scene_min,
								  const float3_t& scene_max,
								  const float3_t& shading_point,
								  const float3_t& view_direction,
								  const float3_t& shading_normal,
								  NeuralImportanceSamplingMLP::InputLayer& input)
{
	float3_t normalized_shading_point =
		make_float3((shading_point.x - scene_min.x) / (scene_max.x - scene_min.x), (shading_point.y - scene_min.y) / (scene_max.y - scene_min.y),
					(shading_point.z - scene_min.z) / (scene_max.z - scene_min.z));

	encode_nisml_position_grid(position_learnable_dense_grid, normalized_shading_point, input.input);

	encode_spherical_harmonics_degree_4(view_direction, input.input + NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE);

	constexpr unsigned int NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE = NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE + NIS_VIEW_DIRECTION_ENCODED_SIZE;
	encode_one_blob<NIS_NORMAL_ONE_BLOB_BIN_COUNT, NIS_NORMAL_ONE_BLOB_KERNEL>(0.5f * (shading_normal.x + 1.0f),
																			   input.input + NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE);
	encode_one_blob<NIS_NORMAL_ONE_BLOB_BIN_COUNT, NIS_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.y + 1.0f), input.input + NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE + NIS_NORMAL_ONE_BLOB_BIN_COUNT);
	encode_one_blob<NIS_NORMAL_ONE_BLOB_BIN_COUNT, NIS_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.z + 1.0f), input.input + NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE + 2 * NIS_NORMAL_ONE_BLOB_BIN_COUNT);
}

HIPRT_DEVICE void build_nis_log_baseline_weights(const HIPRTRenderData& render_data,
												 const NISMLDevice& neural_light_sampling,
												 float3_t shading_point,
												 float3_t view_direction,
												 float3_t shading_normal,
												 float sg_specular_weight,
												 float alpha_x,

												 float alpha_y,
												 float* log_baseline_weights)
{
	const unsigned int invalid_node_index = 0xFFFFFFFF;
	unsigned int cluster_count			  = hippt::min(neural_light_sampling.cluster_count, static_cast<unsigned int>(NIS_MAX_CLUSTER_COUNT));

	for (unsigned int cluster_index = 0; cluster_index < NIS_MAX_CLUSTER_COUNT; cluster_index++)
		log_baseline_weights[cluster_index] = -INFINITY;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		unsigned int node_index = neural_light_sampling.cluster_node_indices[cluster_index];
		if (node_index == invalid_node_index)
			continue;

		float importance = light_tree_sg_node_importance(render_data.light_tree_sg.nodes[node_index], spec_data, shading_point, view_direction, shading_normal,
														 sg_specular_weight, alpha_x, alpha_y);
		if (importance > 0.0f)
			log_baseline_weights[cluster_index] = logf(importance);
	}
}

HIPRT_DEVICE bool evaluate_nis_softmax(const float* log_baseline_weights, const float* residuals, unsigned int cluster_count, float* probabilities)
{
	cluster_count = hippt::min(cluster_count, static_cast<unsigned int>(NIS_MAX_CLUSTER_COUNT));
	for (unsigned int cluster_index = 0; cluster_index < NIS_MAX_CLUSTER_COUNT; cluster_index++)
		probabilities[cluster_index] = 0.0f;

	float maximum_combined_logit = -INFINITY;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		if (log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		float combined_logit = log_baseline_weights[cluster_index] + residuals[cluster_index];
		if (combined_logit > maximum_combined_logit)
			maximum_combined_logit = combined_logit;
	}

	if (maximum_combined_logit == -INFINITY)
		return false;

	float exponential_denominator = 0.0f;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		if (log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		float combined_logit = log_baseline_weights[cluster_index] + residuals[cluster_index];
		exponential_denominator += expf(combined_logit - maximum_combined_logit);
	}

	if (!(exponential_denominator > 0.0f))
		return false;

	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		if (log_baseline_weights[cluster_index] == -INFINITY)
			continue;

		float combined_logit		 = log_baseline_weights[cluster_index] + residuals[cluster_index];
		probabilities[cluster_index] = expf(combined_logit - maximum_combined_logit) / exponential_denominator;
	}

	return true;
}

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
	float probabilities[NIS_MAX_CLUSTER_COUNT];

	if (!evaluate_nis_softmax(cluster_log_baseline_weights, residuals, cluster_count, probabilities))
	{
		out_cluster_probability = 0.0f;

		return cluster_count;
	}

	float random_value			 = rng();
	float cumulative_probability = 0.0f;
	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		cumulative_probability += probabilities[cluster_index];
		if (random_value < cumulative_probability && probabilities[cluster_index] > 0.0f)
		{
			out_cluster_probability = probabilities[cluster_index];

			return cluster_index;
		}
	}

	for (unsigned int cluster_index = cluster_count; cluster_index > 0u; cluster_index--)
		if (probabilities[cluster_index - 1u] > 0.0f)
		{
			out_cluster_probability = probabilities[cluster_index - 1u];

			return cluster_index - 1u;
		}

	out_cluster_probability = 0.0f;

	return cluster_count;
}

HIPRT_DEVICE float evaluate_nis_cluster_probability(const NISMLDevice& neural_light_sampling, const float* residuals, unsigned int target_cluster_index)
{
	unsigned int cluster_count				  = neural_light_sampling.cluster_count;
	const float* cluster_log_baseline_weights = neural_light_sampling.cluster_log_baseline_weights;
	float probabilities[NIS_MAX_CLUSTER_COUNT];

	if (target_cluster_index >= cluster_count || !evaluate_nis_softmax(cluster_log_baseline_weights, residuals, cluster_count, probabilities))
		return 0.0f;

	return probabilities[target_cluster_index];
}

HIPRT_DEVICE unsigned int infer_and_sample_nis_cluster(const NISMLDevice& neural_light_sampling,
													   Xorshift32Generator& rng,
													   const float3_t& shading_point,
													   const float3_t& view_direction,
													   const float3_t& shading_normal,
													   const HIPRTRenderData& render_data,
													   float& out_cluster_probability)
{
	NeuralImportanceSamplingMLP::InputLayer input;
	build_nis_input(render_data.nis_ml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max, shading_point,
					view_direction, shading_normal, input);

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
	NeuralImportanceSamplingMLP::InputLayer input;
	build_nis_input(render_data.nis_ml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max, shading_point,
					view_direction, shading_normal, input);

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
	NISMLDevice neural_light_sampling = render_data.nis_ml;

	const LightTreeSGDevice& light_tree = render_data.light_tree_sg;

	unsigned int invalid_node_index = 0xFFFFFFFF;
	unsigned int cluster_count		= neural_light_sampling.cluster_count;

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
	build_nis_log_baseline_weights(render_data, neural_light_sampling, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y,
								   cluster_log_baseline_weights);
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
		cluster_log_baseline_weights[cluster_position] = importance > 0.0f ? logf(importance) : -INFINITY;
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
	sampled_light.cluster_probability			 = cluster_probability;
	sampled_light.conditional_light_probability	 = conditional_sample.pdf;
	sampled_light.emissive_triangle_pdf			 = cluster_probability * conditional_sample.pdf;
	if (!(sampled_light.emissive_triangle_pdf > 0.0f))
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

	unsigned int invalid_node_index	  = 0xFFFFFFFF;
	unsigned int invalid_cluster_slot = 0xFF;

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
	build_nis_log_baseline_weights(render_data, neural_light_sampling, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y,
								   cluster_log_baseline_weights);
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
		cluster_log_baseline_weights[cluster_position] = importance > 0.0f ? logf(importance) : -INFINITY;
	}

	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;
	float cluster_probability =
		infer_nis_cluster_probability(neural_light_sampling, target_cluster_index, shading_point, view_direction, shading_normal, render_data);
	if (!(cluster_probability > 0.0f))
		return 0.0f;

	unsigned int bit_trail = light_tree.bit_trails[global_emissive_triangle_index];
	if (bit_trail == invalid_node_index)
		return 0.0f;

	unsigned int current_node_index = target_cluster_node_index;
	unsigned int current_depth		= neural_light_sampling.cluster_node_depths[target_cluster_index];

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

	return final_pdf > 0.0f ? final_pdf : 0.0f;
}

#endif
