/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PATH_TRACING_H
#define DEVICE_INCLUDES_PATH_TRACING_H

#include "Device/includes/AABBRasterize.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSurfaceNormalFace.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/LightSampling/NEEDeferredMISContext.h"
#include "Device/includes/LightSampling/NISML/NISML.h"
#include "Device/includes/RussianRoulette.h"

#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "HostDeviceCommon/KernelOptions/SSBNPermutationOptions.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE unsigned int illumination_aware_kd_tree_debug_cell_normal_face_key(const HIPRTRenderData& render_data, unsigned int pixel_index)
{
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	float3_t primary_hit			= render_data.g_buffer.primary_hit_position[pixel_index];
	unsigned int guiding_cell_index = render_data.kd_tree_device.core.find_guiding_cell(primary_hit);
	if (guiding_cell_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	float3_t shading_normal	 = render_data.g_buffer.shading_normals[pixel_index].unpack();
	unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(shading_normal);

	// The key identifies one of the six normal-face entries owned by a guiding cell.
	return guiding_cell_index * static_cast<unsigned int>(SurfaceNormalFace_Count) + normal_face + 1u;
}

HIPRT_DEVICE ColorRGB32F path_tracing_nisml_debug_heatmap(float normalized_value)
{
	normalized_value = hippt::clamp(0.0f, 1.0f, normalized_value);

	if (normalized_value < 0.5f)
	{
		float interpolation = normalized_value * 2.0f;
		return ColorRGB32F(0.0f, interpolation, 1.0f - interpolation);
	}

	float interpolation = (normalized_value - 0.5f) * 2.0f;
	return ColorRGB32F(interpolation, 1.0f - interpolation, 0.0f);
}

HIPRT_DEVICE bool path_tracing_compute_nisml_entropy_debug_value(const HIPRTRenderData& render_data, int pixel_index, float& out_debug_value)
{
#if NISMLDebugMode != NISML_DEBUG_MODE_ENTROPY
	return false;
#else
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		return false;

	NISMLDevice neural_light_sampling = render_data.nisml;
	unsigned int cluster_count		  = neural_light_sampling.cluster_count;
	if (cluster_count == 0 || cluster_count > NISML_MAX_CLUSTER_COUNT || neural_light_sampling.cluster_node_indices == nullptr ||
		neural_light_sampling.position_learnable_dense_grid.features_fp16 == nullptr)
		return false;

	float3_t shading_point	= render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
	float3_t shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();

	DeviceUnpackedEffectiveMaterial material = render_data.g_buffer.materials[pixel_index].unpack();
	float sg_specular_weight;
	float alpha_x;
	float alpha_y;
	get_sg_specular_importance_parameters(material, sg_specular_weight, alpha_x, alpha_y);

	float cluster_log_baseline_weights[NISML_MAX_CLUSTER_COUNT];
	build_nisml_log_baseline_weights(render_data, neural_light_sampling, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y,
									 cluster_log_baseline_weights);

	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;

	NeuralImportanceSamplingMLP::InputLayer input;
	build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
					  shading_point, view_direction, shading_normal, input);

	float residuals[NISML_MAX_CLUSTER_COUNT];
	neural_light_sampling.mlp.inference_single_thread(input, residuals);

	if (!evaluate_nisml_softmax(cluster_log_baseline_weights, residuals, cluster_count))
		return false;

	float probability_sum = 0.0f;
	float entropy		  = 0.0f;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		float probability = cluster_log_baseline_weights[cluster_index];
		if (!(probability > 0.0f))
			continue;

		probability_sum += probability;

		// Shannon entropy H = -sum(p_i * log(p_i)) for i in [0, K-1] where K is the number of clusters and p_i is the probability of cluster i
		entropy -= probability * logf(probability);
	}

	if (!(probability_sum > 0.0f))
		return false;

	if (cluster_count <= 1)
		out_debug_value = 0.0f;
	else
		out_debug_value = entropy / logf(static_cast<float>(cluster_count));

	return true;
#endif
}

HIPRT_DEVICE bool path_tracing_compute_nisml_kl_divergence_debug_value(const HIPRTRenderData& render_data, int pixel_index, float& out_debug_value)
{
#if NISMLDebugMode != NISML_DEBUG_MODE_KL_DIVERGENCE
	return false;
#else
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		return false;

	NISMLDevice neural_light_sampling = render_data.nisml;
	unsigned int cluster_count		  = neural_light_sampling.cluster_count;
	if (cluster_count == 0 || cluster_count > NISML_MAX_CLUSTER_COUNT || neural_light_sampling.cluster_node_indices == nullptr ||
		neural_light_sampling.position_learnable_dense_grid.features_fp16 == nullptr)
		return false;

	float3_t shading_point	= render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
	float3_t shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();

	DeviceUnpackedEffectiveMaterial material = render_data.g_buffer.materials[pixel_index].unpack();
	float sg_specular_weight;
	float alpha_x;
	float alpha_y;
	get_sg_specular_importance_parameters(material, sg_specular_weight, alpha_x, alpha_y);

	float cluster_log_baseline_weights[NISML_MAX_CLUSTER_COUNT];
	build_nisml_log_baseline_weights(render_data, neural_light_sampling, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y,
									 cluster_log_baseline_weights);

	float baseline_probabilities[NISML_MAX_CLUSTER_COUNT];
	for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
		baseline_probabilities[cluster_index] = cluster_log_baseline_weights[cluster_index];

	float zero_residuals[NISML_MAX_CLUSTER_COUNT] = {};
	if (!evaluate_nisml_softmax(baseline_probabilities, zero_residuals, cluster_count))
		return false;

	neural_light_sampling.cluster_log_baseline_weights = cluster_log_baseline_weights;

	NeuralImportanceSamplingMLP::InputLayer input;
	build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
					  shading_point, view_direction, shading_normal, input);

	float residuals[NISML_MAX_CLUSTER_COUNT];
	neural_light_sampling.mlp.inference_single_thread(input, residuals);

	if (!evaluate_nisml_softmax(cluster_log_baseline_weights, residuals, cluster_count))
		return false;

	float kl_divergence = 0.0f;
	for (unsigned int cluster_index = 0; cluster_index < cluster_count; cluster_index++)
	{
		float neural_probability = cluster_log_baseline_weights[cluster_index];
		if (!(neural_probability > 0.0f))
			continue;

		float baseline_probability = baseline_probabilities[cluster_index];
		if (!(baseline_probability > 0.0f))
			return false;

		kl_divergence += neural_probability * logf(neural_probability / baseline_probability);
	}

	out_debug_value = 1.0f - hippt::intrin_expf(-kl_divergence);

	return true;
#endif
}

HIPRT_DEVICE bool path_tracing_compute_nisml_latent_activation_color(const HIPRTRenderData& render_data, int pixel_index, ColorRGB32F& out_debug_color)
{
#if NISMLDebugMode != NISML_DEBUG_MODE_LATENT_ACTIVATIONS
	return false;
#else
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		return false;

	NISMLDevice neural_light_sampling = render_data.nisml;
	if (neural_light_sampling.position_learnable_dense_grid.features_fp16 == nullptr)
		return false;

	float3_t shading_point	= render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
	float3_t shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();

	NeuralImportanceSamplingMLP::InputLayer input;
	build_nisml_input(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
					  shading_point, view_direction, shading_normal, input);

	float latent_activations[NeuralImportanceSamplingMLP::HIDDEN_LAYER_SIZE];
	unsigned int latent_layer_index = NeuralImportanceSamplingMLP::HIDDEN_LAYER_COUNT - 1;
	if (!neural_light_sampling.mlp.inference_single_thread_hidden_layer(input, latent_layer_index, latent_activations))
		return false;

	float activation_norm_squared = 0.0f;
	for (unsigned int neuron = 0; neuron < NeuralImportanceSamplingMLP::HIDDEN_LAYER_SIZE; neuron++)
		activation_norm_squared += latent_activations[neuron] * latent_activations[neuron];

	if (!(activation_norm_squared > 1.0e-12f))
		return false;

	float activation_norm = hippt::sqrt(activation_norm_squared);
	float projections[3]  = {};
	for (unsigned int projection_channel = 0; projection_channel < 3; projection_channel++)
	{
		float projection = 0.0f;
		for (unsigned int neuron = 0; neuron < NeuralImportanceSamplingMLP::HIDDEN_LAYER_SIZE; neuron++)
		{
			unsigned int hash = (neuron + 1u) * 0x9E3779B9u;
			hash ^= (projection_channel + 1u) * 0x85EBCA6Bu;
			hash ^= hash >> 16;
			hash *= 0x7FEB352Du;
			hash ^= hash >> 15;

			float projection_sign = (hash & 1u) == 0 ? 1.0f : -1.0f;
			projection += latent_activations[neuron] / activation_norm * projection_sign;
		}

		projections[projection_channel] = projection * 0.125f;
	}

	float quantized_color[3];
	for (unsigned int color_channel = 0; color_channel < 3; color_channel++)
	{
		float color_channel_value	   = hippt::clamp(0.0f, 1.0f, 0.5f + 0.5f * projections[color_channel]);
		unsigned int quantized_channel = static_cast<unsigned int>(color_channel_value * 15.0f + 0.5f);
		quantized_color[color_channel] = static_cast<float>(quantized_channel) / 15.0f;
	}

	out_debug_color =
		ColorRGB32F::random_color(static_cast<unsigned int>(quantized_color[0] * 0xFFFFFFFF) + static_cast<unsigned int>(quantized_color[1] * 0xFFFFFFFF) +
								  static_cast<unsigned int>(quantized_color[2] * 0xFFFFFFFF));

	return true;
#endif
}

HIPRT_DEVICE bool path_tracing_find_indirect_bounce_intersection(
	HIPRTRenderData& render_data, hiprtRay ray, RayPayload& out_ray_payload, HitInfo& out_closest_hit_info, Xorshift32Generator& random_number_generator)
{
	return trace_main_path_ray(render_data, ray, out_ray_payload, out_closest_hit_info, out_closest_hit_info.primitive_index, random_number_generator);
}

/**
 * If sampleDirectionOnly is 'true', only the direction for the next bounce will be computed
 * but without evaluating the contribution of the BSDF or the PDF.
 */
template <bool sampleDirectionOnly = false>
HIPRT_DEVICE void path_tracing_sample_bsdf_next_indirect_bounce(HIPRTRenderData& render_data,
																RayPayload& ray_payload,
																const HitInfo& closest_hit_info,
																const float3_t view_direction,
																ColorRGB32F& out_bsdf_color,
																float3_t& out_bounce_direction,
																float& out_bsdf_pdf,
																Xorshift32Generator& random_number_generator,
																BSDFIncidentLightInfo& out_sampled_light_info)
{
	BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f),
							 out_sampled_light_info, ray_payload.volume_state, true, ray_payload.material, ray_payload.accumulated_roughness);

	out_bsdf_color = bsdf_dispatcher_sample<sampleDirectionOnly>(render_data, bsdf_context, out_bounce_direction, out_bsdf_pdf, random_number_generator);

	ray_payload.accumulate_roughness(out_sampled_light_info);
}

/**
 * Returns the new ray throughput after attenuation of the given 'current_throughput'
 */
HIPRT_DEVICE ColorRGB32F path_tracing_update_ray_throughput(HIPRTRenderData& render_data,
															RayPayload& ray_payload,
															const HitInfo& closest_hit_info,
															ColorRGB32F current_throughput,
															float& rr_throughput_scaling,
															ColorRGB32F bsdf_color_cos_theta,
															float3_t bounce_direction,
															float bsdf_pdf,
															Xorshift32Generator& random_number_generator,
															NEEDeferredMISContext& nee_deferred_MIS_context,
															bool apply_russian_roulette = true)
{
	ColorRGB32F throughput_attenuation = bsdf_color_cos_theta / bsdf_pdf;

	// Russian roulette
	if (apply_russian_roulette && !do_russian_roulette(render_data.render_settings, ray_payload.bounce, current_throughput, rr_throughput_scaling,
													   throughput_attenuation, random_number_generator))
		return ColorRGB32F(0.0f);

	// Dispersion ray throughput filter
	current_throughput *= get_dispersion_ray_color(ray_payload.volume_state.sampled_wavelength, ray_payload.material.dispersion_scale);
	current_throughput *= throughput_attenuation;
	// Clamp every component to a minimum of 1.0e-5f to avoid numerical instabilities that can
	// happen: with some material, the throughput can get so low that it becomes denormalized and
	// this can cause issues in some parts of the renderer (most notably the NaN detection)
	current_throughput.max(ColorRGB32F(1.0e-5f, 1.0e-5f, 1.0e-5f));

	ray_payload.next_ray_state = RayState::BOUNCE;

	return current_throughput;
}

/**
 * Returns the new ray throughput after attenuation of the given 'current_throughput'
 */
HIPRT_DEVICE ColorRGB32F path_tracing_update_ray_throughput(HIPRTRenderData& render_data,
															RayPayload& ray_payload,
															const HitInfo& closest_hit_info,
															ColorRGB32F current_throughput,
															ColorRGB32F bsdf_color_cos_theta,
															float3_t bounce_direction,
															float bsdf_pdf,
															Xorshift32Generator& random_number_generator,
															NEEDeferredMISContext& nee_deferred_MIS_context,
															bool apply_russian_roulette = true)
{
	// TODO this function with the unused_rr_throughput_scaling is unused?
	float unused_rr_throughput_scaling;
	return path_tracing_update_ray_throughput(render_data, ray_payload, closest_hit_info, current_throughput, unused_rr_throughput_scaling,
											  bsdf_color_cos_theta, bounce_direction, bsdf_pdf, random_number_generator, nee_deferred_MIS_context,
											  apply_russian_roulette);
}

/**
 * Returns true if the bounce was sampled successfully,
 * false otherwise (is the BSDF sample failed, if russian roulette killed the sample, ...)
 *
 * If sampleDirectionOnly is 'true', only the direction for the next bounce will be computed
 * but without evaluating the contribution of the BSDF or the PDF.
 */
HIPRT_DEVICE bool path_tracing_compute_next_indirect_bounce(HIPRTRenderData& render_data,
															RayPayload& ray_payload,
															HitInfo& closest_hit_info,
															float3_t view_direction,
															hiprtRay& out_ray,
															Xorshift32Generator& random_number_generator,
															BSDFIncidentLightInfo& incident_light_info,
															NEEDeferredMISContext& nee_deferred_MIS_context)
{
	nee_deferred_MIS_context.fill_last_hit_information(closest_hit_info, view_direction, ray_payload.volume_state, ray_payload.material,
													   ray_payload.throughput);

	ColorRGB32F bsdf_color;
	float3_t bounce_direction;
	float bsdf_pdf;
	path_tracing_sample_bsdf_next_indirect_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf,
												  random_number_generator, incident_light_info);
	ColorRGB32F bsdf_color_cos_theta = bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal));

	nee_deferred_MIS_context.fill_last_bsdf_information(bsdf_color_cos_theta, bsdf_pdf);

	out_ray.origin	  = closest_hit_info.inter_point;
	out_ray.direction = bounce_direction;

	// Terminate ray if bad sampling
	if (bsdf_pdf <= 0.0f)
		return false;

	ray_payload.throughput = path_tracing_update_ray_throughput(render_data, ray_payload, closest_hit_info, ray_payload.throughput, bsdf_color_cos_theta,
																bounce_direction, bsdf_pdf, random_number_generator, nee_deferred_MIS_context);
	if (ray_payload.throughput.is_black())
		// Killed by russian roulette
		return false;

	return true;
}

HIPRT_DEVICE void store_denoiser_AOVs(HIPRTRenderData& render_data, uint32_t pixel_index, float3_t shading_normal, ColorRGB32F base_color)
{
	if (render_data.render_settings.sample_number == 0)
		render_data.aux_buffers.denoiser_albedo[pixel_index] = base_color;
	else
		render_data.aux_buffers.denoiser_albedo[pixel_index] =
			(render_data.aux_buffers.denoiser_albedo[pixel_index] * render_data.render_settings.denoiser_AOV_accumulation_counter + base_color) /
			(render_data.render_settings.denoiser_AOV_accumulation_counter + 1.0f);

	if (render_data.render_settings.sample_number == 0)
		render_data.aux_buffers.denoiser_normals[pixel_index] = shading_normal;
	else
	{
		float3_t accumulated_normal =
			(render_data.aux_buffers.denoiser_normals[pixel_index] * static_cast<float>(render_data.render_settings.denoiser_AOV_accumulation_counter) +
			 shading_normal) /
			(render_data.render_settings.denoiser_AOV_accumulation_counter + 1.0f);
		float normal_length = hippt::length(accumulated_normal);
		if (!hippt::is_zero(normal_length))
			// Checking that it is non-zero otherwise we would accumulate a persistent NaN in the buffer when normalizing by the 0-length
			render_data.aux_buffers.denoiser_normals[pixel_index] = accumulated_normal / normal_length;
	}
}

HIPRT_DEVICE ColorRGB32F path_tracing_miss_gather_envmap(HIPRTRenderData& render_data,
														 const ColorRGB32F& ray_throughput,
														 float3_t ray_direction,
														 int bounce,
														 uint32_t pixel_index,
														 ColorRGB32F* out_raw_envmap_emission = nullptr)
{
	ColorRGB32F skysphere_color;

	if (render_data.world_settings.ambient_light_type == AmbientLightType::UNIFORM || render_data.bsdfs_data.white_furnace_mode)
		skysphere_color = render_data.world_settings.uniform_light_color;
	else if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP && render_data.world_settings.envmap_intensity == 0.0f)
		return ColorRGB32F(0.0f);
	else if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
	{
#if EnvmapSamplingStrategy != ESS_NO_SAMPLING
		// If we have sampling, only taking envmap into account on camera ray miss
		if (bounce == 0)
#endif
		{
			// We're only getting the skysphere radiance for the first rays because the
			// syksphere is importance sampled.
			skysphere_color = eval_envmap_no_pdf(render_data.world_settings, ray_direction);

#if EnvmapSamplingStrategy == ESS_NO_SAMPLING
			// If we don't have envmap sampling, we're only going to unscale on
			// bounce 0 (which is when a ray misses directly --> background color).
			// Otherwise, if not bounce 2, we do want to take the scaling into
			// account so this if will fail and the envmap color will never be unscaled
			if (!render_data.world_settings.envmap_scale_background_intensity && bounce == 0)
#else
			if (!render_data.world_settings.envmap_scale_background_intensity)
#endif
				// Un-scaling the envmap if the user doesn't want to scale the background
				skysphere_color /= (render_data.world_settings.envmap_intensity * render_data.world_settings.envmap_packed_scaling_factor);
		}
	}

	skysphere_color = clamp_light_contribution(skysphere_color, render_data.render_settings.envmap_contribution_clamp, /* clamp condition */ true);
	if (out_raw_envmap_emission)
		*out_raw_envmap_emission = skysphere_color;

	ColorRGB32F indirect_lighting_contribution = skysphere_color * ray_throughput;
	// Only clamping with the indirect lighting clamp value if
	// this is bounce > 0 (thanks to /* clamp condition */ bounce > 0)
	ColorRGB32F clamped_indirect_lighting_contribution =
		clamp_light_contribution(indirect_lighting_contribution, render_data.render_settings.indirect_contribution_clamp,
								 /* clamp condition */ bounce > 0);

	if (bounce == 0)
		// The camera ray missed so we don't have the normals but we have the base color
		store_denoiser_AOVs(render_data, pixel_index, make_float3(0, 0, 0), skysphere_color);

	return clamped_indirect_lighting_contribution;
}

HIPRT_DEVICE ColorRGB32F path_tracing_miss_gather_envmap(HIPRTRenderData& render_data, RayPayload& ray_payload, float3_t ray_direction, uint32_t pixel_index)
{
	return path_tracing_miss_gather_envmap(render_data, ray_payload.throughput, ray_direction, ray_payload.bounce, pixel_index);
}

#define DEFAULT_DEBUG_COLOR ColorRGB32F(-42.0f, -42.0f, -42.0f)

HIPRT_DEVICE bool path_tracing_sample_is_in_subset(const HIPRTRenderSettings& render_settings)
{
	if (render_settings.sample_subset_min == 0 && render_settings.sample_subset_max == 0)
		return true;

	if (render_settings.sample_subset_min < 0 || render_settings.sample_subset_max < render_settings.sample_subset_min)
		return false;

	return render_settings.sample_number >= static_cast<unsigned int>(render_settings.sample_subset_min) &&
		   render_settings.sample_number <= static_cast<unsigned int>(render_settings.sample_subset_max);
}

HIPRT_DEVICE unsigned int path_tracing_number_of_samples_in_subset(const HIPRTRenderSettings& render_settings)
{
	if (render_settings.sample_subset_min == 0 && render_settings.sample_subset_max == 0)
		return render_settings.sample_number + 1;

	if (render_settings.sample_subset_min < 0 || render_settings.sample_subset_max < render_settings.sample_subset_min ||
		render_settings.sample_number < static_cast<unsigned int>(render_settings.sample_subset_min))
		return 0;

	unsigned int number_of_samples = render_settings.sample_number - static_cast<unsigned int>(render_settings.sample_subset_min) + 1;
	unsigned int subset_size	   = static_cast<unsigned int>(render_settings.sample_subset_max - render_settings.sample_subset_min + 1);

	return hippt::min(number_of_samples, subset_size);
}

HIPRT_DEVICE void path_tracing_accumulate_color(const HIPRTRenderData& render_data,
												uint32_t pixel_index,
												const ColorRGB32F& ray_color,
												const ColorRGB32F& debug_color = DEFAULT_DEBUG_COLOR)
{
	render_data.buffers.last_frame_ray_colors[pixel_index] = ray_color;

	const bool sample_is_in_subset						= path_tracing_sample_is_in_subset(render_data.render_settings);
	const unsigned int number_of_samples_in_subset		= path_tracing_number_of_samples_in_subset(render_data.render_settings);
	const unsigned int number_of_samples_before_current = sample_is_in_subset ? number_of_samples_in_subset - 1 : number_of_samples_in_subset;

	if (sample_is_in_subset)
	{
		if (debug_color != DEFAULT_DEBUG_COLOR)
			render_data.buffers.accumulated_ray_colors[pixel_index] = debug_color;
		else if (number_of_samples_before_current == 0)
			render_data.buffers.accumulated_ray_colors[pixel_index] = ray_color * (render_data.render_settings.sample_number + 1);
		else
		{
			// The framebuffer is divided by the global sample count when it is displayed. Recover the sum of the selected
			// samples from the previous framebuffer value before adding the current sample.
			ColorRGB32F accumulated_subset_sum = render_data.buffers.accumulated_ray_colors[pixel_index] /
												 static_cast<float>(render_data.render_settings.sample_number) * number_of_samples_before_current;
			ColorRGB32F accumulated_subset_average = (accumulated_subset_sum + ray_color) / static_cast<float>(number_of_samples_in_subset);

			render_data.buffers.accumulated_ray_colors[pixel_index] = accumulated_subset_average * (render_data.render_settings.sample_number + 1);
		}
	}
	else
	{
		if (number_of_samples_before_current == 0)
			render_data.buffers.accumulated_ray_colors[pixel_index] = ColorRGB32F();
		else
			// Keep the selected-sample average unchanged while the display divisor keeps increasing.
			render_data.buffers.accumulated_ray_colors[pixel_index] = render_data.buffers.accumulated_ray_colors[pixel_index] /
																	  static_cast<float>(render_data.render_settings.sample_number) *
																	  (render_data.render_settings.sample_number + 1);
	}

	if (sample_is_in_subset && render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		float squared_luminance_of_samples = ray_color.luminance() * ray_color.luminance();
		// We can only use these buffers if the adaptive sampling or the stop noise threshold is enabled.
		// Otherwise, the buffers are destroyed to save some VRAM so they are not accessible
		render_data.aux_buffers.pixel_squared_luminance[pixel_index] += squared_luminance_of_samples;
	}

	if (sample_is_in_subset && render_data.buffers.gmon_estimator.sets != nullptr)
	{
		// GMoN is in use, accumulating in the GMoN sets

		unsigned int offset = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y *
								  render_data.buffers.gmon_estimator.next_set_to_accumulate +
							  pixel_index;

		if (number_of_samples_before_current == 0)
			render_data.buffers.gmon_estimator.sets[offset] = ray_color;
		else
			render_data.buffers.gmon_estimator.sets[offset] += ray_color;
	}
}

HIPRT_DEVICE bool path_tracing_pixel_is_on_tree_cut_bounding_box_edge(const HIPRTRenderData& render_data, int pixel_index, unsigned int& out_box_index)
{
	const LightTreeSGSettings& settings = render_data.light_tree_sg.settings;
	unsigned int* tree_cut_node_indices = render_data.light_tree_sg.tree_cut_node_indices;
	unsigned int tree_cut_size			= settings.effective_tree_cut_size;

	if (tree_cut_node_indices == nullptr || render_data.light_tree_sg.nodes == nullptr || tree_cut_size == 0)
		return false;

	for (unsigned int tree_cut_node_index = 0; tree_cut_node_index < tree_cut_size; tree_cut_node_index++)
	{
		const unsigned int node_index = tree_cut_node_indices[tree_cut_node_index];
		if (node_index == 0xFFFFFFFF)
			continue;

		const LightTreeSGNodeDevice& node = render_data.light_tree_sg.nodes[node_index];
		if (aabb_rasterize_pixel_is_on_aabb_edge(render_data.current_camera, render_data.render_settings.render_resolution, pixel_index, node.bounds_min,
												 node.bounds_max))
		{
			out_box_index = tree_cut_node_index;

			return true;
		}
	}

	return false;
}

HIPRT_DEVICE void path_tracing_compute_debug_view_debug_color(
	const HIPRTRenderData& render_data, RayPayload& ray_payload, int pixel_index, Xorshift32Generator& rng, ColorRGB32F& out_debug_color)
{
	out_debug_color = DEFAULT_DEBUG_COLOR;

	// Modifying the ray color such that we display some debug color to the screen

#if DirectLightNEEPlusPlusDisplayShadowRaysDiscarded == KERNEL_OPTION_TRUE
	// Nothing to do, the debug is already handled in the shadow ray NEE function
#elif NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// We have a first hit
		float3_t primary_hit	= render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
		float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

		unsigned int trash_checksum;
		NEEPlusPlusContext context;
		context.envmap		   = false;
		context.point_on_light = make_float3(0, 0, 0);
		context.shaded_point   = primary_hit;

		out_debug_color = ColorRGB32F::random_color(render_data.nee_plus_plus.hash_context(context, render_data.current_camera, trash_checksum));
		out_debug_color *= (render_data.render_settings.sample_number + 1);
		out_debug_color *= hippt::dot(shading_normal, view_direction);
	}
#elif ReGIRDebugMode != REGIR_DEBUG_MODE_NO_DEBUG
#if ReGIRDebugMode == REGIR_DEBUG_MODE_GRID_CELLS
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// We have a first hit
		float3_t primary_hit		= render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal				= render_data.g_buffer.geometric_normals[pixel_index].unpack();
		float3_t view_direction		= render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
		float primary_hit_roughness = render_data.g_buffer.materials[pixel_index].get_roughness();

		out_debug_color =
			render_data.render_settings.regir_settings.get_random_cell_color(primary_hit, normal, render_data.current_camera, primary_hit_roughness, true);
		out_debug_color *= (render_data.render_settings.sample_number + 1);
		out_debug_color *= hippt::dot(normal, view_direction);
	}
#elif ReGIRDebugMode == REGIR_DEBUG_MODE_AVERAGE_CELL_NON_CANONICAL_RESERVOIR_CONTRIBUTION
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

		unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(primary_hit);

		float average_contribution = 0.0f;
		for (int i = 0; i < render_data.render_settings.regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell(); i++)
		{
			ReGIRReservoir reservoir = render_data.render_settings.regir_settings.get_cell_non_canonical_reservoir_from_cell_reservoir_index(cell_index, i);
			average_contribution += reservoir.sample.target_function * reservoir.UCW;
		}

		// Averaging
		average_contribution /= render_data.render_settings.regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell();
		// Scaling by the debug factor for visualization purposes
		average_contribution *= render_data.render_settings.regir_settings.debug_view_scale_factor;
		// Scaling by SPP
		average_contribution *= (render_data.render_settings.sample_number + 1);

		out_debug_color = ColorRGB32F(average_contribution);
	}
#elif ReGIRDebugMode == REGIR_DEBUG_MODE_AVERAGE_CELL_CANONICAL_RESERVOIR_CONTRIBUTION
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

		unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(primary_hit);

		float average_contribution = 0.0f;
		for (int i = 0; i < render_data.render_settings.regir_settings.grid_fill.get_canonical_reservoir_count_per_cell(); i++)
		{
			ReGIRReservoir reservoir = render_data.render_settings.regir_settings.get_cell_canonical_reservoir_from_cell_reservoir_index(cell_index, i);
			average_contribution += reservoir.sample.target_function * reservoir.UCW;
		}

		// Averaging
		average_contribution /= render_data.render_settings.regir_settings.grid_fill.get_canonical_reservoir_count_per_cell();
		// Scaling by the debug factor for visualization purposes
		average_contribution *= render_data.render_settings.regir_settings.debug_view_scale_factor;
		// Scaling by SPP
		average_contribution *= (render_data.render_settings.sample_number + 1);

		out_debug_color = ColorRGB32F(average_contribution);
	}
#elif ReGIRDebugMode == REGIR_DEBUG_MODE_REPRESENTATIVE_POINTS
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		float3_t primary_hit		= render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal				= render_data.g_buffer.geometric_normals[pixel_index].unpack();
		float primary_hit_roughness = render_data.g_buffer.materials[pixel_index].get_roughness();

		unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(
			primary_hit, normal, render_data.current_camera, primary_hit_roughness, true);

		ColorRGB32F color;
		float3_t rep_point = ReGIR_get_cell_world_point(render_data, cell_index, true);
		// Interpreting debug_view_scale_factor as a distance
		if (hippt::length(rep_point - primary_hit) < render_data.render_settings.regir_settings.debug_view_scale_factor)
			color = ColorRGB32F::random_color(cell_index + 1);

		// Scaling by SPP so that the visualization doesn't get darker and darker with increasing number of SPP
		color *= render_data.render_settings.sample_number + 1;

		out_debug_color = ColorRGB32F(color);
	}
#elif ReGIRDebugMode == REGIR_DEBUG_MODE_REPRESENTATIVE_NORMALS
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		float3_t primary_hit		= render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal				= render_data.g_buffer.geometric_normals[pixel_index].unpack();
		float primary_hit_roughness = render_data.g_buffer.materials[pixel_index].get_roughness();

		unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(
			primary_hit, normal, render_data.current_camera, primary_hit_roughness, true);

		ColorRGB32F color = (ColorRGB32F(ReGIR_get_cell_world_normal(render_data, cell_index, true)) + ColorRGB32F(1.0f)) * 0.5f;

		// Scaling by SPP so that the visualization doesn't get darker and darker with increasing number of SPP
		color *= render_data.render_settings.sample_number + 1;

		out_debug_color = ColorRGB32F(color);
	}
#endif // ReGIR debug mode

#elif NISMLDebugMode != NISML_DEBUG_MODE_NO_DEBUG && ILLUMINATION_AWARE_KD_TREE_IS_NISML(DirectLightNEEEstimator, DirectLightSamplingStrategy)
#if NISMLDebugMode == NISML_DEBUG_MODE_LATENT_ACTIVATIONS
	ColorRGB32F nisml_latent_activation_color;
	if (path_tracing_compute_nisml_latent_activation_color(render_data, pixel_index, nisml_latent_activation_color))
		out_debug_color = nisml_latent_activation_color * (render_data.render_settings.sample_number + 1);

#elif NISMLDebugMode == NISML_DEBUG_MODE_ENTROPY
	float nisml_debug_value;
	if (path_tracing_compute_nisml_entropy_debug_value(render_data, pixel_index, nisml_debug_value))
		out_debug_color = path_tracing_nisml_debug_heatmap(nisml_debug_value) * (render_data.render_settings.sample_number + 1);

#elif NISMLDebugMode == NISML_DEBUG_MODE_KL_DIVERGENCE
	float nisml_debug_value;
	if (path_tracing_compute_nisml_kl_divergence_debug_value(render_data, pixel_index, nisml_debug_value))
		out_debug_color = path_tracing_nisml_debug_heatmap(nisml_debug_value) * (render_data.render_settings.sample_number + 1);
#endif // NISML debug mode

#elif IlluminationAwareKDTreeDebugMode != ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_NO_DEBUG &&                                                                    \
	ILLUMINATION_AWARE_KD_TREE_IS_ENABLED(DirectLightNEEEstimator, DirectLightSamplingStrategy)
#if IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// We have a first hit
		float3_t primary_hit			= render_data.g_buffer.primary_hit_position[pixel_index];
		unsigned int guiding_cell_index = render_data.kd_tree_device.core.find_guiding_cell(primary_hit);

		if (guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// A cell outline is detected where a neighboring primary-hit pixel belongs to a different guiding cell.
		const unsigned int guiding_cell_index = render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		const unsigned int image_width		  = render_data.render_settings.render_resolution.x;
		const unsigned int image_height		  = render_data.render_settings.render_resolution.y;
		const unsigned int pixel_x			  = pixel_index % image_width;
		const unsigned int pixel_y			  = pixel_index / image_width;
		bool is_cell_outline				  = false;

		if (pixel_x > 0)
		{
			const unsigned int neighbor_pixel_index = pixel_index - 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_x + 1 < image_width)
		{
			const unsigned int neighbor_pixel_index = pixel_index + 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_y > 0)
		{
			const unsigned int neighbor_pixel_index = pixel_index - image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_y + 1 < image_height)
		{
			const unsigned int neighbor_pixel_index = pixel_index + image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (is_cell_outline && guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_NORMAL_SOLID
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		unsigned int cell_normal_face_key = illumination_aware_kd_tree_debug_cell_normal_face_key(render_data, pixel_index);

		if (cell_normal_face_key != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(cell_normal_face_key) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_NORMAL_OUTLINE
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		unsigned int cell_normal_face_key = illumination_aware_kd_tree_debug_cell_normal_face_key(render_data, pixel_index);
		unsigned int image_width		  = render_data.render_settings.render_resolution.x;
		unsigned int image_height		  = render_data.render_settings.render_resolution.y;
		unsigned int pixel_x			  = pixel_index % image_width;
		unsigned int pixel_y			  = pixel_index / image_width;
		bool is_cell_normal_face_outline  = false;

		if (cell_normal_face_key != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		{
			if (pixel_x > 0)
			{
				unsigned int neighbor_pixel_index = pixel_index - 1;
				if (illumination_aware_kd_tree_debug_cell_normal_face_key(render_data, neighbor_pixel_index) != cell_normal_face_key)
					is_cell_normal_face_outline = true;
			}

			if (pixel_x + 1 < image_width)
			{
				unsigned int neighbor_pixel_index = pixel_index + 1;
				if (illumination_aware_kd_tree_debug_cell_normal_face_key(render_data, neighbor_pixel_index) != cell_normal_face_key)
					is_cell_normal_face_outline = true;
			}

			if (pixel_y > 0)
			{
				unsigned int neighbor_pixel_index = pixel_index - image_width;
				if (illumination_aware_kd_tree_debug_cell_normal_face_key(render_data, neighbor_pixel_index) != cell_normal_face_key)
					is_cell_normal_face_outline = true;
			}

			if (pixel_y + 1 < image_height)
			{
				unsigned int neighbor_pixel_index = pixel_index + image_width;
				if (illumination_aware_kd_tree_debug_cell_normal_face_key(render_data, neighbor_pixel_index) != cell_normal_face_key)
					is_cell_normal_face_outline = true;
			}
		}

		if (is_cell_normal_face_outline)
			out_debug_color = ColorRGB32F::random_color(cell_normal_face_key) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE_AND_LOOKAHEAD
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// A cell outline is detected where a neighboring primary-hit pixel belongs to a different guiding cell.
		unsigned int guiding_cell_index	  = render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		unsigned int lookahead_cell_index = render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		unsigned int image_width		  = render_data.render_settings.render_resolution.x;
		unsigned int image_height		  = render_data.render_settings.render_resolution.y;
		unsigned int pixel_x			  = pixel_index % image_width;
		unsigned int pixel_y			  = pixel_index / image_width;

		bool is_cell_outline		   = false;
		bool is_lookahead_cell_outline = false;
		if (pixel_x > 0)
		{
			unsigned int neighbor_pixel_index = pixel_index - 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_x + 1 < image_width)
		{
			unsigned int neighbor_pixel_index = pixel_index + 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_y > 0)
		{
			unsigned int neighbor_pixel_index = pixel_index - image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_y + 1 < image_height)
		{
			unsigned int neighbor_pixel_index = pixel_index + image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (is_cell_outline && guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
		else if (is_lookahead_cell_outline && lookahead_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			// Using the same color as the encompassing guiding cell but darker
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1) * 0.5f;
	}
#endif // LightTreeSG debug mode

#elif SSBNPermutationDebugHashGrid == KERNEL_OPTION_TRUE
	ColorRGB32F color = ColorRGB32F::random_color(render_data.ssbn_settings.screen_space_hash_grid[pixel_index].x);
	color *= render_data.render_settings.sample_number + 1;

	out_debug_color = ColorRGB32F(color);
#elif SSBNPermutationDebugSeeds == KERNEL_OPTION_TRUE
	ColorRGB32F color = ColorRGB32F(render_data.get_input_random_seed(pixel_index) / (float)((unsigned int)(-1)));
	color *= render_data.render_settings.sample_number + 1;

	out_debug_color = ColorRGB32F(color);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_GRID_CELLS && ReSTIRPGEnable == KERNEL_OPTION_TRUE
	float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();

	ColorRGB32F color;

	unsigned int checksum;
	unsigned int cell_index =
		render_data.render_settings.restir_pg_settings.get_hash_grid_cell_index_from_position_data(primary_hit, normal, render_data.current_camera, checksum) %
		render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells;
	if (!HashGrid::resolve_collision<ReSTIRPGHashGridCollisionResolveSteps, false>(
			render_data.render_settings.restir_pg_settings.hash_grid_checksums, render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells,
			cell_index, checksum))
		color = ColorRGB32F();
	else
		color = ColorRGB32F::random_color(cell_index);

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_DIRECTION && ReSTIRPGEnable == KERNEL_OPTION_TRUE
	ColorRGB32F color;

	if (render_data.render_settings.sample_number == 0 || render_data.render_settings.nb_bounces == 0)
	{
		// At sample 0 we don't have the distributions yet so we can't fetch the directions from the distributions themselves but we can just display the
		// directions that the directions are initialized with which are directions on the fibonacci sphere
		color = ColorRGB32F(fibonacci_sphere_direction(render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number,
													   ReSTIRPGDistributionComponentCount))
					.abs();
	}
	else
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		ReSTIRPGDistribution distribution =
			render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(primary_hit, normal, render_data.current_camera);

		if (distribution.distribution_components[0].weight == 0.0f)
			color = ColorRGB32F(0.0f);
		else if (!hippt::is_finite(distribution.distribution_components[0].weight))
			color = ColorRGB32F(1.0e10f, 0.0e10f, 1.0e10f);
		else
			color = ColorRGB32F(
						hippt::normalize(
							distribution.distribution_components[render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number]
								.vmf.axis))
						.abs();
	}

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_SHARPNESS && ReSTIRPGEnable == KERNEL_OPTION_TRUE
	ColorRGB32F color;

	if (render_data.render_settings.sample_number == 0)
		// At sample 0 all distributions are at sharpness 50
		color = ColorRGB32F(50.0f);
	else
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		ReSTIRPGDistribution distribution =
			render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(primary_hit, normal, render_data.current_camera);

		color = ColorRGB32F(distribution.distribution_components[render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number]
								.vmf.sharpness)
					.abs();
	}

	color /= render_data.render_settings.restir_pg_settings.debug_normalization_factor;

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_WEIGHT && ReSTIRPGEnable == KERNEL_OPTION_TRUE
	ColorRGB32F color;

	if (render_data.render_settings.sample_number == 0)
		// At sample 0 all distributions are at sharpness 50
		color = ColorRGB32F(0.25f);
	else
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		ReSTIRPGDistribution distribution =
			render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(primary_hit, normal, render_data.current_camera);

		color = ColorRGB32F(
					distribution.distribution_components[render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number].weight)
					.abs();
	}

	color *= render_data.render_settings.restir_pg_settings.debug_normalization_factor;

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#endif // Switch on the debugging option

	// Draw the SG tree cut bounding boxes last so they remain visible on top of any other debug view.
	unsigned int box_index = -1;
	if (render_data.light_tree_sg.settings.debug_draw_tree_cut_bounding_boxes &&
		path_tracing_pixel_is_on_tree_cut_bounding_box_edge(render_data, pixel_index, box_index))
	{
		if (render_data.light_tree_sg.settings.debug_draw_random_colors_boxes)
			out_debug_color = ColorRGB32F::random_color(box_index) * (render_data.render_settings.sample_number + 1);
		else
			out_debug_color = ColorRGB32F(2.0f, 0.0f, 0.0f) * (render_data.render_settings.sample_number + 1);
	}
}

#endif
