/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDE_REGIR_FINAL_SHADING_H
#define DEVICE_INCLUDE_REGIR_FINAL_SHADING_H

#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightUtils.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_ReGIR(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    if (!MaterialUtils::can_do_light_sampling(ray_payload.material))
        return ColorRGB32F(0.0f);

    bool point_outside_grid = false;

	ReGIRShadingAdditionalInfo additional_infos;
    LightSampleInformation light_sample = sample_one_emissive_triangle_regir_with_info(render_data,
        closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
        closest_hit_info.primitive_index, ray_payload, 
        point_outside_grid, random_number_generator, additional_infos);

    if (!point_outside_grid)
    {
        if (light_sample.area_measure_pdf <= 0.0f)
            // Can happen for very small triangles
            return ColorRGB32F(0.0f);

#if ReGIR_ShadingResamplingShadeAllSamples == KERNEL_OPTION_TRUE
        // If we're shading all samples, we already have the perfectly computed
        // radiance in additional_infos so we can just return that
        return additional_infos.sample_radiance;
#endif
        // ReGIR succeeded with sampling, just shooting a shadow ray to validate visibility

        float3 shadow_ray_origin = closest_hit_info.inter_point;
        float3 shadow_ray_direction = light_sample.point_on_light - shadow_ray_origin;
        float distance_to_light = hippt::length(shadow_ray_direction);
        float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;
    
        hiprtRay shadow_ray;
        shadow_ray.origin = shadow_ray_origin;
        shadow_ray.direction = shadow_ray_direction_normalized;
    
        // NEE++ context for the shadow ray
        NEEPlusPlusContext nee_plus_plus_context;
        nee_plus_plus_context.point_on_light = light_sample.point_on_light;
        nee_plus_plus_context.shaded_point = shadow_ray_origin;
        nee_plus_plus_context.shaded_point_surface_normal = closest_hit_info.shading_normal;

        bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator, ray_payload.bounce);

        if (!in_shadow)
            return additional_infos.sample_radiance / light_sample.area_measure_pdf / nee_plus_plus_context.unoccluded_probability;
        else
            return ColorRGB32F(0.0f);
    }
    else
    {
        #if ReGIR_FallbackLightSamplingStrategy == LSS_BASE_REGIR
        // Invalid fallback strategy
        invalid ReGIR light sampling fallback strategy
        #endif
        
        // Fallback method as the point was outside of the ReGIR grid
        light_sample = sample_one_emissive_triangle<ReGIR_FallbackLightSamplingStrategy>(render_data,
            closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
            closest_hit_info.primitive_index, ray_payload,
            random_number_generator);
         
        float3 shadow_ray_origin = closest_hit_info.inter_point;
        float3 shadow_ray_direction = light_sample.point_on_light - shadow_ray_origin;
        float distance_to_light = hippt::length(shadow_ray_direction);
        float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

        hiprtRay shadow_ray;
        shadow_ray.origin = shadow_ray_origin;
        shadow_ray.direction = shadow_ray_direction_normalized;

        // NEE++ context for the shadow ray
        NEEPlusPlusContext nee_plus_plus_context;
        nee_plus_plus_context.point_on_light = light_sample.point_on_light;
        nee_plus_plus_context.shaded_point = shadow_ray_origin;
        nee_plus_plus_context.shaded_point_surface_normal = closest_hit_info.shading_normal;

        ColorRGB32F light_source_radiance;
        // abs() here to allow backfacing light sources
        float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

        if (dot_light_source > 0.0f)
        {
            bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator, ray_payload.bounce);

            if (!in_shadow)
            {
                float bsdf_pdf;

                BSDFIncidentLightInfo incident_light_info = light_sample.incident_light_info;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
                BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
                BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif
                ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

                if (bsdf_pdf != 0.0f)
                {
                    // Conversion to solid angle from surface area measure
                    float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
                    if (light_sample_solid_angle_pdf > 0.0f)
                    {
                        float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));
                        light_source_radiance = light_sample.emission * cosine_term * bsdf_color / light_sample_solid_angle_pdf / nee_plus_plus_context.unoccluded_probability;

                        // Just a CPU-only sanity check
                        sanity_check</* CPUOnly */ true>(render_data, light_source_radiance, 0, 0);

                        return light_source_radiance;
                    }
                }
            }
        }
    }

    return ColorRGB32F(0.0f);
}

#endif
