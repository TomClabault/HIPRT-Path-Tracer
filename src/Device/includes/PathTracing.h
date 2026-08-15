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
			ColorRGB32F accumulated_subset_sum	   = render_data.buffers.accumulated_ray_colors[pixel_index] /
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

#endif
