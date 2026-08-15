/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NISML_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NISML_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSamplingCommon.h"
#include "Device/includes/Neural/NISML/NISMLPositionLearnableDenseGrid.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
#include "HostDeviceCommon/Maths/VecTypes.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

#include <math.h>

struct NISMLLightSample
{
	int emissive_triangle_global_index = -1;

	unsigned int cluster_index = 0;

	float cluster_probability			= 0.0f;
	float conditional_light_probability = 0.0f;
	float emissive_triangle_pdf			= 0.0f;
};

HIPRT_DEVICE void build_nisml_input(const NISMLPositionLearnableDenseGridDevice& position_learnable_dense_grid,
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

	constexpr unsigned int NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE = NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE + NISML_VIEW_DIRECTION_ENCODED_SIZE;
	encode_one_blob<NISML_NORMAL_ONE_BLOB_BIN_COUNT, NISML_NORMAL_ONE_BLOB_KERNEL>(0.5f * (shading_normal.x + 1.0f),
																				   input.input + NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE);
	encode_one_blob<NISML_NORMAL_ONE_BLOB_BIN_COUNT, NISML_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.y + 1.0f), input.input + NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE + NISML_NORMAL_ONE_BLOB_BIN_COUNT);
	encode_one_blob<NISML_NORMAL_ONE_BLOB_BIN_COUNT, NISML_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.z + 1.0f), input.input + NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE + 2 * NISML_NORMAL_ONE_BLOB_BIN_COUNT);
}

HIPRT_DEVICE void load_nisml_input_wmma(const NISMLPositionLearnableDenseGridDevice& position_learnable_dense_grid,
										const float3_t& scene_min,
										const float3_t& scene_max,
										const float3_t& shading_point,
										const float3_t& view_direction,
										const float3_t& shading_normal,
										fp16* activations_buffer,
										unsigned int thread_index)
{
	constexpr unsigned int ACTIVATION_STRIDE = NeuralImportanceSamplingMLP::BLOCK_SIZE;

	float3_t normalized_shading_point =
		make_float3((shading_point.x - scene_min.x) / (scene_max.x - scene_min.x), (shading_point.y - scene_min.y) / (scene_max.y - scene_min.y),
					(shading_point.z - scene_min.z) / (scene_max.z - scene_min.z));

	encode_nisml_position_grid_wmma(position_learnable_dense_grid, normalized_shading_point, activations_buffer, ACTIVATION_STRIDE, thread_index);

	encode_spherical_harmonics_degree_4_wmma(view_direction, activations_buffer, NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE, ACTIVATION_STRIDE,
											 thread_index);

	constexpr unsigned int NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE = NISML_POSITION_LEARNABLE_DENSE_GRID_ENCODED_SIZE + NISML_VIEW_DIRECTION_ENCODED_SIZE;

	encode_one_blob_wmma<NISML_NORMAL_ONE_BLOB_BIN_COUNT, NISML_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.x + 1.0f), activations_buffer, NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE, ACTIVATION_STRIDE, thread_index);

	encode_one_blob_wmma<NISML_NORMAL_ONE_BLOB_BIN_COUNT, NISML_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.y + 1.0f), activations_buffer, NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE + NISML_NORMAL_ONE_BLOB_BIN_COUNT, ACTIVATION_STRIDE,
		thread_index);

	encode_one_blob_wmma<NISML_NORMAL_ONE_BLOB_BIN_COUNT, NISML_NORMAL_ONE_BLOB_KERNEL>(
		0.5f * (shading_normal.z + 1.0f), activations_buffer, NISML_POSITION_AND_VIEW_DIR_ENCODED_SIZE + 2 * NISML_NORMAL_ONE_BLOB_BIN_COUNT, ACTIVATION_STRIDE,
		thread_index);

	for (unsigned int input_index = NISML_INPUT_SIZE_ENCODED; input_index < NeuralImportanceSamplingMLP::INPUT_SIZE_PADDED_WMMA; input_index++)
		activations_buffer[input_index * ACTIVATION_STRIDE + thread_index] = static_cast<fp16>(0.0f);
}

HIPRT_DEVICE void build_nisml_log_baseline_weights(const HIPRTRenderData& render_data,
												   const NISMLDevice& neural_light_sampling,
												   float3_t shading_point,
												   float3_t view_direction,
												   float3_t shading_normal,
												   float sg_specular_weight,
												   float alpha_x,

												   float alpha_y,
												   float* log_baseline_weights,
												   unsigned int output_stride = 1,
												   unsigned int output_index  = 0)
{
	unsigned int cluster_count = hippt::min(neural_light_sampling.cluster_count, static_cast<unsigned int>(NISML_MAX_CLUSTER_COUNT));

#if NISMLUseSGImportancesKDTreeCaches == KERNEL_OPTION_TRUE
	const IlluminationAwareKDTreeDevice& kd_tree_device = render_data.kd_tree_device;
	if (kd_tree_device.core.nodes != nullptr && kd_tree_device.nisml.nisml_cache != nullptr && kd_tree_device.nisml.nisml_cache_ready != nullptr)
	{
		unsigned int node_index = kd_tree_device.core.find_guiding_cell(shading_point);
		if (node_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX && node_index < kd_tree_device.core.node_capacity)
		{
			unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(shading_normal);
			unsigned int cache_index = kd_tree_device.nisml.get_nisml_cache_index(node_index, normal_face);
			if (kd_tree_device.nisml.nisml_cache_ready[cache_index] != 0)
			{
				for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
					log_baseline_weights[cluster_index * output_stride + output_index] =
						kd_tree_device.nisml.nisml_cache[cache_index].log_importances[cluster_index];

				return;
			}
		}
	}
#endif

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	constexpr unsigned int invalid_node_index = 0xFFFFFFFF;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		unsigned int node_index = neural_light_sampling.cluster_node_indices[cluster_index];
		if (node_index == invalid_node_index)
			continue;

		float importance = light_tree_sg_node_importance(render_data.light_tree_sg.nodes[node_index], spec_data, shading_point, view_direction, shading_normal,
														 sg_specular_weight, alpha_x, alpha_y);

		log_baseline_weights[cluster_index * output_stride + output_index] = hippt::intrin_logf(hippt::max(1.0e-5f, importance));
	}
}

template <typename residual_type>
HIPRT_DEVICE bool evaluate_nisml_softmax(float* in_out_log_baseline_weights_probabilities,
										 const residual_type* residuals,
										 unsigned int cluster_count,
										 unsigned int output_stride	  = 1,
										 unsigned int output_index	  = 0,
										 unsigned int residual_stride = 1,
										 unsigned int residual_index  = 0)
{
	cluster_count = hippt::min(cluster_count, static_cast<unsigned int>(NISML_MAX_CLUSTER_COUNT));

	float maximum_combined_logit = -INFINITY;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		unsigned int output_value_index = cluster_index * output_stride + output_index;
		if (in_out_log_baseline_weights_probabilities[output_value_index] == -INFINITY)
			continue;

		float combined_logit =
			in_out_log_baseline_weights_probabilities[output_value_index] + static_cast<float>(residuals[cluster_index * residual_stride + residual_index]);
		maximum_combined_logit = hippt::max(maximum_combined_logit, combined_logit);
	}

	if (maximum_combined_logit == -INFINITY)
		return false;

	float exponential_denominator = 0.0f;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		unsigned int output_value_index = cluster_index * output_stride + output_index;
		if (in_out_log_baseline_weights_probabilities[output_value_index] == -INFINITY)
			continue;

		float combined_logit =
			in_out_log_baseline_weights_probabilities[output_value_index] + static_cast<float>(residuals[cluster_index * residual_stride + residual_index]);
		float exp_value = hippt::intrin_expf(combined_logit - maximum_combined_logit);
		// Storing in the buffer so we don't have to recompute the exponentials in the next loop
		in_out_log_baseline_weights_probabilities[output_value_index] = exp_value;

		exponential_denominator += exp_value;
	}

	if (!(exponential_denominator > 0.0f))
		return false;

	float exponential_denominator_reciprocal = 1.0f / exponential_denominator;

	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		unsigned int output_value_index = cluster_index * output_stride + output_index;
		if (in_out_log_baseline_weights_probabilities[output_value_index] == -INFINITY)
		{
			in_out_log_baseline_weights_probabilities[output_value_index] = 0.0f;

			continue;
		}

		in_out_log_baseline_weights_probabilities[output_value_index] *= exponential_denominator_reciprocal;
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

HIPRT_DEVICE unsigned int sample_nisml_cluster(const NISMLDevice& neural_light_sampling,
											   const float* residuals,
											   Xorshift32Generator& rng,
											   float& out_cluster_probability)
{
	unsigned int cluster_count							 = neural_light_sampling.cluster_count;
	float* cluster_log_baseline_weights_or_probabilities = neural_light_sampling.cluster_log_baseline_weights;

	if (!evaluate_nisml_softmax(cluster_log_baseline_weights_or_probabilities, residuals, cluster_count))
	{
		out_cluster_probability = 0.0f;

		return cluster_count;
	}

	float random_value			 = rng();
	float cumulative_probability = 0.0f;
	for (unsigned int cluster_index = 0u; cluster_index < cluster_count; cluster_index++)
	{
		cumulative_probability += cluster_log_baseline_weights_or_probabilities[cluster_index];
		if (random_value < cumulative_probability && cluster_log_baseline_weights_or_probabilities[cluster_index] > 0.0f)
		{
			out_cluster_probability = cluster_log_baseline_weights_or_probabilities[cluster_index];

			return cluster_index;
		}
	}

	for (unsigned int cluster_index = cluster_count; cluster_index > 0u; cluster_index--)
	{
		if (cluster_log_baseline_weights_or_probabilities[cluster_index - 1u] > 0.0f)
		{
			out_cluster_probability = cluster_log_baseline_weights_or_probabilities[cluster_index - 1u];

			return cluster_index - 1u;
		}
	}

	out_cluster_probability = 0.0f;

	return cluster_count;
}

HIPRT_DEVICE float evaluate_nisml_cluster_probability(const NISMLDevice& neural_light_sampling, const float* residuals, unsigned int target_cluster_index)
{
	unsigned int cluster_count							 = neural_light_sampling.cluster_count;
	float* cluster_log_baseline_weights_or_probabilities = neural_light_sampling.cluster_log_baseline_weights;

	if (target_cluster_index >= cluster_count || !evaluate_nisml_softmax(cluster_log_baseline_weights_or_probabilities, residuals, cluster_count))
		return 0.0f;

	return cluster_log_baseline_weights_or_probabilities[target_cluster_index];
}

HIPRT_DEVICE unsigned int infer_and_sample_nisml_cluster(const NISMLDevice& neural_light_sampling,
														 Xorshift32Generator& rng,
														 const float3_t& shading_point,
														 const float3_t& view_direction,
														 const float3_t& shading_normal,
														 const HIPRTRenderData& render_data,
														 float& out_cluster_probability)
{
	NeuralImportanceSamplingMLP::InputLayer input;
	build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
					  shading_point, view_direction, shading_normal, input);

	float residuals[NISML_MAX_CLUSTER_COUNT];
	neural_light_sampling.mlp.inference_single_thread(input, residuals);

	return sample_nisml_cluster(neural_light_sampling, residuals, rng, out_cluster_probability);
}

HIPRT_DEVICE float infer_nisml_cluster_probability(const NISMLDevice& neural_light_sampling,
												   unsigned int target_cluster_index,
												   const float3_t& shading_point,
												   const float3_t& view_direction,
												   const float3_t& shading_normal,
												   const HIPRTRenderData& render_data)
{
	NeuralImportanceSamplingMLP::InputLayer input;
	build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
					  shading_point, view_direction, shading_normal, input);

	float residuals[NISML_MAX_CLUSTER_COUNT];
	neural_light_sampling.mlp.inference_single_thread(input, residuals);

	return evaluate_nisml_cluster_probability(neural_light_sampling, residuals, target_cluster_index);
}

HIPRT_DEVICE NISMLLightSample sample_one_emissive_triangle_neural_many_lights(const HIPRTRenderData& render_data,
																			  const float3_t& shading_point,
																			  const float3_t& view_direction,
																			  const float3_t& shading_normal,
																			  const DeviceUnpackedEffectiveMaterial& material,
																			  Xorshift32Generator& random_number_generator)
{
	NISMLLightSample sampled_light;
	NISMLDevice neural_light_sampling = render_data.nisml;

	const LightTreeSGDevice& light_tree = render_data.light_tree_sg;

	unsigned int invalid_node_index = 0xFFFFFFFF;
	unsigned int cluster_count		= neural_light_sampling.cluster_count;

	if (cluster_count == 0 || cluster_count > NISML_MAX_CLUSTER_COUNT || light_tree.nodes == nullptr || neural_light_sampling.cluster_node_indices == nullptr)
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

	float cluster_log_baseline_weights[NISML_MAX_CLUSTER_COUNT];
	build_nisml_log_baseline_weights(render_data, neural_light_sampling, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y,
									 cluster_log_baseline_weights);

	// TODO stop these shenanigans and just pass cluster_log_baseline_weights down
	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;
	float cluster_probability						   = 0.0f;
	unsigned int selected_cluster_position = infer_and_sample_nisml_cluster(neural_light_sampling, random_number_generator, shading_point, view_direction,
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
		return NISMLLightSample();

	return sampled_light;
}

HIPRT_DEVICE float pdf_of_emissive_triangle_nis(const HIPRTRenderData& render_data,
												float3_t shading_point,
												float3_t view_direction,
												float3_t shading_normal,
												const DeviceUnpackedEffectiveMaterial& material,
												int global_emissive_triangle_index)
{
	NISMLDevice neural_light_sampling	= render_data.nisml;
	const LightTreeSGDevice& light_tree = render_data.light_tree_sg;

	unsigned int invalid_node_index	  = 0xFFFFFFFF;
	unsigned int invalid_cluster_slot = 0xFF;

	if (global_emissive_triangle_index < 0 || neural_light_sampling.cluster_count == 0 || neural_light_sampling.cluster_count > NISML_MAX_CLUSTER_COUNT ||
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

	float cluster_log_baseline_weights[NISML_MAX_CLUSTER_COUNT];
	build_nisml_log_baseline_weights(render_data, neural_light_sampling, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y,
									 cluster_log_baseline_weights);

	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;
	float cluster_probability =
		infer_nisml_cluster_probability(neural_light_sampling, target_cluster_index, shading_point, view_direction, shading_normal, render_data);
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
