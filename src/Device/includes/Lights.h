/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHTS_H
#define DEVICE_LIGHTS_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/DI/FinalShading.h"
#include "Device/includes/RIS/RIS.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_no_MIS(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    ColorRGB32F light_source_radiance;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
    if (!(light_sample_pdf > 0.0f))
        // Can happen for very small triangles
        return ColorRGB32F(0.0f);

    float3 shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
    float3 shadow_ray_direction = random_light_point - shadow_ray_origin;
    float distance_to_light = hippt::length(shadow_ray_direction);
    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = shadow_ray_origin;
    shadow_ray.direction = shadow_ray_direction_normalized;

    // abs() here to allow backfacing light sources
    float dot_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -shadow_ray.direction));
    if (dot_light_source > 0.0f)
    {
        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

        if (!in_shadow)
        {
            float brdf_pdf;
            RayVolumeState trash_volume_state = ray_payload.volume_state;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, brdf_pdf);
            if (brdf_pdf != 0.0f)
            {
                // Conversion to solid angle from surface area measure
                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= dot_light_source;

                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance = light_source_info.emission * cosine_term * bsdf_color / light_sample_pdf;
            }
        }
    }

    return light_source_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_bsdf(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;

    ColorRGB32F bsdf_radiance = ColorRGB32F(0.0f);

    float direction_pdf;
    float3 sampled_brdf_direction;
    RayVolumeState trash_volume_state = ray_payload.volume_state;
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);

    bool refraction_sampled = hippt::dot(sampled_brdf_direction, closest_hit_info.shading_normal * inside_surface_multiplier) < 0;
    if (direction_pdf > 0.0f)
    {
        hiprtRay new_ray;
        new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
        new_ray.direction = sampled_brdf_direction;

        if (refraction_sampled)
        {
            // If we sampled a refraction, we're pushing the origin of the shadow ray "through"
            // the surface so that our ray can refract inside the surface
            //
            // If we don't do that and we just use the 'evaluated_point' computed at the very
            // beginning of the function, we're going to intersect ourselves

            new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier * -1.0f;
        }

        ShadowLightRayHitInfo shadow_light_ray_hit_info;
        bool inter_found = evaluate_shadow_light_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info);

        // Checking that we did hit something and if we hit something,
        // it needs to be emissive
        if (inter_found && !shadow_light_ray_hit_info.hit_emission.is_black())
        {
            float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            bsdf_radiance = bsdf_color * cosine_term * shadow_light_ray_hit_info.hit_emission / direction_pdf;
        }
    }

    return bsdf_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_MIS(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    float light_sample_pdf;
    ColorRGB32F light_source_radiance_mis;
    LightSourceInformation light_source_info;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
    if (light_sample_pdf <= 0.0f)
        // Can happen for very small triangles
        return ColorRGB32F(0.0f);

    float3 shadow_ray_direction = random_light_point - evaluated_point;
    float distance_to_light = hippt::length(shadow_ray_direction);
    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = evaluated_point;
    shadow_ray.direction = shadow_ray_direction_normalized;

    // abs() here to allow backfacing light sources
    float dot_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -shadow_ray.direction));
    if (dot_light_source > 0.0f)
    {
        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

        if (!in_shadow)
        {
            float bsdf_pdf;
            RayVolumeState trash_volume_state = ray_payload.volume_state;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, bsdf_pdf);
            if (bsdf_pdf != 0.0f)
            {
                // Conversion to solid angle from surface area measure
                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= dot_light_source;

                float mis_weight = balance_heuristic(light_sample_pdf, bsdf_pdf);

                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance_mis = bsdf_color * cosine_term * light_source_info.emission * mis_weight / light_sample_pdf;
            }
        }
    }

    ColorRGB32F bsdf_radiance_mis;

    float direction_pdf;
    float3 sampled_bsdf_direction;
    float3 bsdf_shadow_ray_origin = evaluated_point;
    RayVolumeState trash_volume_state = ray_payload.volume_state;
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_bsdf_direction, direction_pdf, random_number_generator);
    bool refraction_sampled = hippt::dot(sampled_bsdf_direction, closest_hit_info.shading_normal * inside_surface_multiplier) < 0;
    if (refraction_sampled)
    {
        // If we sampled a refraction, we're pushing the origin of the shadow ray "through"
        // the surface so that our ray can refract inside the surface
        //
        // If we don't do that and we just use the 'evaluated_point' computed at the very
        // beginning of the function, we're going to intersect ourselves

        bsdf_shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier * -1.0f;
    }

    if (direction_pdf > 0)
    {
        hiprtRay new_ray;
        new_ray.origin = bsdf_shadow_ray_origin;
        new_ray.direction = sampled_bsdf_direction;

        ShadowLightRayHitInfo shadow_light_ray_hit_info;
        bool inter_found = evaluate_shadow_light_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info);

        // Checking that we did hit something and if we hit something,
        // it needs to be emissive
        if (inter_found && !shadow_light_ray_hit_info.hit_emission.is_black())
        {
            float light_pdf = pdf_of_emissive_triangle_hit(render_data, shadow_light_ray_hit_info, sampled_bsdf_direction);
            float mis_weight = balance_heuristic(direction_pdf, light_pdf);

            // Using abs here because we want the dot product to be positive.
            // You may be thinking that if we're doing this, then we're not going to discard BSDF
            // sampled direction that are below the surface (whereas we should discard them).
            // That would be correct but bsdf_dispatcher_sample return a PDF == 0.0f if a bad
            // direction was sampled and if the PDF is 0.0f, we never get to this line of code
            // you're reading. If we are here, this is because we sampled a direction that is
            // correct for the BSDF. Even if the direction is correct, the dot product may be
            // negative in the case of refractions / total internal reflections and so in this case,
            // we'll need to negative the dot product for it to be positive
            float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_bsdf_direction));
            //float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            bsdf_radiance_mis = bsdf_color * cosine_term * shadow_light_ray_hit_info.hit_emission * mis_weight / direction_pdf;
        }
    }

    // Because we're sampling only 1 light out of all the lights of the
    // scene, the probability of having chosen that light is: 1 / numberOfLights
    // This must be factored in the PDF of sampling that light which means that we must
    // divide by 1 / numberOfLights => multiply by numberOfLights
    return light_source_radiance_mis + bsdf_radiance_mis;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, int2 pixel_coords, int2 resolution, int bounce)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        // No emmisive geometry in the scene to sample
        return ColorRGB32F(0.0f);

    if (ray_payload.material.emission.r != 0.0f || ray_payload.material.emission.g != 0.0f || ray_payload.material.emission.b != 0.0f)
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return ColorRGB32F(0.0f);

    ColorRGB32F direct_light_contribution;
#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING
    direct_light_contribution = ColorRGB32F(0.0f);
#elif DirectLightSamplingStrategy == LSS_UNIFORM_ONE_LIGHT
    direct_light_contribution = sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_BSDF
    direct_light_contribution = sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_MIS_LIGHT_BSDF
    direct_light_contribution = sample_one_light_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_RIS_BSDF_AND_LIGHT
    direct_light_contribution = sample_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_RESTIR_DI

    if (bounce == 0)
        // Can only do ReSTIR DI on the first bounce
        direct_light_contribution = sample_light_ReSTIR_DI(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, pixel_coords, resolution);
    else
    {
#if ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
    direct_light_contribution = sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_BSDF
    direct_light_contribution = sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF
    direct_light_contribution = sample_one_light_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT
    direct_light_contribution = sample_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#endif
    }
#endif

    return direct_light_contribution;
}

#endif
