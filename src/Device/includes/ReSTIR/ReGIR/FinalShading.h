/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDE_REGIR_FINAL_SHADING_H
#define DEVICE_INCLUDE_REGIR_FINAL_SHADING_H

#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/TriangleEmissiveSamplingReGIR.h"
#include "Device/includes/Wavefront/WavefrontShadow.h"

#include "HostDeviceCommon/RenderData.h"

template <typename BsdfMaterialType>
HIPRT_DEVICE ColorRGB32F evaluate_regir_fallback_without_visibility(HIPRTRenderData& render_data,
																	RayPayload& ray_payload,
																	BsdfMaterialType& bsdf_material,
																	const HitInfo& closest_hit_info,
																	const float3_t& view_direction,
																	const LightSamplePointInformation& light_sample,
																	float3_t shadow_direction,
																	float distance_to_light,
																	float dot_light_source,
																	Xorshift32Generator& random_number_generator)
{
	float bsdf_pdf;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingStrategy == LSS_BASE_REGIR
	MicrofacetRegularization::RegularizationMode regularization_mode = MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS;
#else
	MicrofacetRegularization::RegularizationMode regularization_mode = MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC;
#endif
	BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
												incident_light_info, ray_payload.volume_state, false, bsdf_material, ray_payload.accumulated_roughness,
												regularization_mode);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
	if (bsdf_pdf == 0.0f)
		return ColorRGB32F(0.0f);

	float light_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
	if (!(light_pdf > 0.0f))
		return ColorRGB32F(0.0f);

	float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));
	return light_sample.emission * cosine_term * bsdf_color / light_pdf / DirectLightIntegrationFactor<DirectLightSamplingStrategy>();
}

template <typename BsdfMaterialType, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE>
HIPRT_DEVICE ColorRGB32F sample_one_light_ReGIR(HIPRTRenderData& render_data,
												RayPayload& ray_payload,
												BsdfMaterialType& bsdf_material,
												const HitInfo closest_hit_info,
												const float3_t& view_direction,
												Xorshift32Generator& random_number_generator,
												WavefrontShadowSink* shadow_sink = nullptr)
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

		ColorRGB32F contribution_if_visible = selected_sample_radiance / light_sample.area_measure_pdf;
		if constexpr (shadow_mode == NEE_SHADOW_WAVEFRONT)
		{
			shadow_sink->emit(shadow_ray.origin, shadow_ray.direction, distance_to_light, closest_hit_info.primitive_index, light_sample.point_on_light,
							  contribution_if_visible, WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS | WAVEFRONT_SHADOW_EMISSIVE_RAW);
			return ColorRGB32F(0.0f);
		}
		else
		{
			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point	 = shadow_ray_origin;
			bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
															   nee_plus_plus_context, random_number_generator);

			return in_shadow ? ColorRGB32F(0.0f) : contribution_if_visible;
		}
#endif // #if ReGIR_ShadingResamplingTargetFunctionVisibility == KERNEL_OPTION_TRUE
	}
	else
	{
#if ReGIRDebugMode == REGIR_DEBUG_MODE_SAMPLING_FALLBACK
		return ColorRGB32F(1.0e10f, 0.0f, 1.0e10f);
#endif

#if ReGIR_FallbackLightSamplingStrategy == LSS_BASE_REGIR
		// Invalid fallback strategy
		invalid ReGIR light sampling fallback strategy
#endif

			// Fallback method as the point was outside of the ReGIR grid
			ColorRGB32F light_source_radiance;

		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
			sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
									  closest_hit_info.geometric_normal, closest_hit_info.primitive_index, ray_payload, random_number_generator);

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
				if constexpr (shadow_mode == NEE_SHADOW_WAVEFRONT)
				{
					ColorRGB32F contribution_if_visible =
						evaluate_regir_fallback_without_visibility(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample,
																   shadow_ray.direction, distance_to_light, dot_light_source, random_number_generator);
					if (!contribution_if_visible.is_black())
						shadow_sink->emit(shadow_ray.origin, shadow_ray.direction, distance_to_light, closest_hit_info.primitive_index,
										  light_sample.point_on_light, contribution_if_visible,
										  WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS | WAVEFRONT_SHADOW_EMISSIVE_RAW);
				}
				else
				{
					bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																	   nee_plus_plus_context, random_number_generator);
					if (!in_shadow)
						light_source_radiance +=
							evaluate_regir_fallback_without_visibility(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample,
																	   shadow_ray.direction, distance_to_light, dot_light_source, random_number_generator);
				}
			}
		}

		return light_source_radiance;
	}

	return ColorRGB32F(0.0f);
}

#endif // #ifndef DEVICE_INCLUDE_REGIR_FINAL_SHADING_H
