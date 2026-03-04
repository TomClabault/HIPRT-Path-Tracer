/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDE_REGIR_FINAL_SHADING_H
#define DEVICE_INCLUDE_REGIR_FINAL_SHADING_H

#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/TriangleEmissiveSamplingReGIR.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ColorRGB32F sample_one_light_ReGIR(HIPRTRenderData& render_data,
												RayPayload& ray_payload,
												const HitInfo closest_hit_info,
												const float3_t& view_direction,
												Xorshift32Generator& random_number_generator)
{
	if (!ray_payload.material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	bool point_outside_grid = false;

	ColorRGB32F selected_sample_radiance;
	LightSamplePointInformation light_sample = sample_one_emissive_triangle_regir_with_selected_sample_radiance(
							render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
							closest_hit_info.primitive_index, ray_payload, point_outside_grid, selected_sample_radiance, random_number_generator);

	if (!point_outside_grid)
	{
		if (light_sample.area_measure_pdf <= 0.0f)
			// Can happen for very small triangles
			return ColorRGB32F(0.0f);

#if ReGIR_ShadingResamplingTargetFunctionVisibility == KERNEL_OPTION_TRUE
		// We already know that a selected sample isn't in shadow otherwise its target
		// function would have been 0 and it would have never been selected
		return selected_sample_radiance / light_sample.area_measure_pdf;
#else
		// ReGIR succeeded with sampling, just shooting a shadow ray to validate visibility
		float3_t shadow_ray_origin				 = closest_hit_info.inter_point;
		float3_t shadow_ray_direction			 = light_sample.point_on_light - shadow_ray_origin;
		float distance_to_light					 = hippt::length(shadow_ray_direction);
		float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

		hiprtRay shadow_ray;
		shadow_ray.origin	 = shadow_ray_origin;
		shadow_ray.direction = shadow_ray_direction_normalized;

		// NEE++ context for the shadow ray
		NEEPlusPlusContext nee_plus_plus_context;
		nee_plus_plus_context.point_on_light = light_sample.point_on_light;
		nee_plus_plus_context.shaded_point	 = shadow_ray_origin;

		bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, nee_plus_plus_context,
														   random_number_generator, ray_payload.bounce);

		if (!in_shadow)
			return selected_sample_radiance / light_sample.area_measure_pdf / nee_plus_plus_context.unoccluded_probability;
		else
			return ColorRGB32F(0.0f);
#endif
	}
	else
	{
#if ReGIR_DebugMode == REGIR_DEBUG_MODE_SAMPLING_FALLBACK
		return ColorRGB32F(1.0e10f, 0.0f, 1.0e10f);
#endif

#if ReGIR_FallbackLightSamplingStrategy == LSS_BASE_REGIR
		// Invalid fallback strategy
		invalid ReGIR light sampling fallback strategy
#endif

								// Fallback method as the point was outside of the ReGIR grid
								ColorRGB32F light_source_radiance;

		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples = sample_one_point_on_light(
								render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
								closest_hit_info.primitive_index, ray_payload, random_number_generator);

		for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
		{
			LightSamplePointInformation& light_sample = light_samples[i];

			if (light_sample.area_measure_pdf <= 0.0f)
				// Can happen for very small triangles or the light
				// sampling technique couldn't sample a triangle
				continue;

			float3_t shadow_ray_origin				 = closest_hit_info.inter_point;
			float3_t shadow_ray_direction			 = light_sample.point_on_light - shadow_ray_origin;
			float distance_to_light					 = hippt::length(shadow_ray_direction);
			float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

			hiprtRay shadow_ray;
			shadow_ray.origin	 = shadow_ray_origin;
			shadow_ray.direction = shadow_ray_direction_normalized;

			// NEE++ context for the shadow ray
			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point	 = shadow_ray_origin;

			// abs() here to allow backfacing light sources
			float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

			if (dot_light_source > 0.0f)
			{
				bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																   nee_plus_plus_context, random_number_generator, ray_payload.bounce);

				if (!in_shadow)
				{
					float bsdf_pdf;

					BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingStrategy == LSS_BASE_REGIR
					BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
											 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
											 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
					BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
											 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
											 MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif
					ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

					if (bsdf_pdf != 0.0f)
					{
						// Conversion to solid angle from surface area measure
						float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
						if (light_sample_solid_angle_pdf > 0.0f)
						{
							float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));
							light_source_radiance += light_sample.emission * cosine_term * bsdf_color / light_sample_solid_angle_pdf /
													 nee_plus_plus_context.unoccluded_probability;

							// Just a CPU-only sanity check
							sanity_check</* CPUOnly */ true>(render_data, light_source_radiance, 0, 0);
						}
					}
				}
			}
		}

		return light_source_radiance / DirectLightIntegrationFactor<DirectLightSamplingStrategy>();
	}

	return ColorRGB32F(0.0f);
}

#endif
