/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PATH_TRACING_H
#define DEVICE_INCLUDES_PATH_TRACING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/LightSampling/NEEDeferredMISContext.h"
#include "Device/includes/RussianRoulette.h"

#include "HostDeviceCommon/KernelOptions/SSBNPermutationOptions.h"
#include "HostDeviceCommon/RenderData.h"

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

HIPRT_DEVICE void path_tracing_accumulate_color(const HIPRTRenderData& render_data,
												uint32_t pixel_index,
												const ColorRGB32F& ray_color,
												const ColorRGB32F& debug_color = DEFAULT_DEBUG_COLOR)
{
	render_data.buffers.last_frame_ray_colors[pixel_index] = ray_color;

#if DisplayOnlySampleN == KERNEL_OPTION_TRUE
	int debug_sample_index = render_data.render_settings.output_debug_sample_N;

	if (render_data.render_settings.sample_number >= debug_sample_index || render_data.render_settings.sample_number == 0)
	{
		if (debug_color == DEFAULT_DEBUG_COLOR)
			render_data.buffers.accumulated_ray_colors[pixel_index] = ray_color * (render_data.render_settings.sample_number + 1);
		else
			render_data.buffers.accumulated_ray_colors[pixel_index] = debug_color;
	}
	else
		render_data.buffers.accumulated_ray_colors[pixel_index] = ColorRGB32F();

#else // DisplayOnlySampleN

	if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
	{
		float squared_luminance_of_samples = ray_color.luminance() * ray_color.luminance();
		// We can only use these buffers if the adaptive sampling or the stop noise threshold is enabled.
		// Otherwise, the buffers are destroyed to save some VRAM so they are not accessible
		render_data.aux_buffers.pixel_squared_luminance[pixel_index] += squared_luminance_of_samples;
	}

	if (debug_color != DEFAULT_DEBUG_COLOR)
		render_data.buffers.accumulated_ray_colors[pixel_index] = debug_color;
	else if (render_data.render_settings.sample_number == 0)
		render_data.buffers.accumulated_ray_colors[pixel_index] = ray_color;
	else
		// If we are at a sample that is not 0, this means that we are accumulating
		render_data.buffers.accumulated_ray_colors[pixel_index] += ray_color;

	if (render_data.buffers.gmon_estimator.sets != nullptr)
	{
		// GMoN is in use, accumulating in the GMoN sets

		unsigned int offset = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y *
								  render_data.buffers.gmon_estimator.next_set_to_accumulate +
							  pixel_index;

		if (render_data.render_settings.sample_number == 0)
			render_data.buffers.gmon_estimator.sets[offset] = ray_color;
		else
			render_data.buffers.gmon_estimator.sets[offset] += ray_color;
	}
#endif
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

#elif IlluminationAwareKDTreeDebugMode != ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_NO_DEBUG && DirectLightNEEEstimator == LSS_SG_TREE_LEARNT_DISTRIBUTIONS
#if IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// We have a first hit
		float3_t primary_hit			= render_data.g_buffer.primary_hit_position[pixel_index];
		unsigned int guiding_cell_index = render_data.illumination_aware_kd_tree.find_guiding_cell(primary_hit);

		if (guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// A cell outline is detected where a neighboring primary-hit pixel belongs to a different guiding cell.
		const unsigned int guiding_cell_index =
			render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		const unsigned int image_width	= render_data.render_settings.render_resolution.x;
		const unsigned int image_height = render_data.render_settings.render_resolution.y;
		const unsigned int pixel_x		= pixel_index % image_width;
		const unsigned int pixel_y		= pixel_index / image_width;
		bool is_cell_outline			= false;

		if (pixel_x > 0)
		{
			const unsigned int neighbor_pixel_index = pixel_index - 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_x + 1 < image_width)
		{
			const unsigned int neighbor_pixel_index = pixel_index + 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_y > 0)
		{
			const unsigned int neighbor_pixel_index = pixel_index - image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_y + 1 < image_height)
		{
			const unsigned int neighbor_pixel_index = pixel_index + image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (is_cell_outline && guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE_AND_LOOKAHEAD
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// A cell outline is detected where a neighboring primary-hit pixel belongs to a different guiding cell.
		unsigned int guiding_cell_index	  = render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		unsigned int lookahead_cell_index = render_data.illumination_aware_kd_tree.find_lookahead_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
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
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.illumination_aware_kd_tree.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_x + 1 < image_width)
		{
			unsigned int neighbor_pixel_index = pixel_index + 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.illumination_aware_kd_tree.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_y > 0)
		{
			unsigned int neighbor_pixel_index = pixel_index - image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.illumination_aware_kd_tree.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_y + 1 < image_height)
		{
			unsigned int neighbor_pixel_index = pixel_index + image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.illumination_aware_kd_tree.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.illumination_aware_kd_tree.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
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
}

#endif
