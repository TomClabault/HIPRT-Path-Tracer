/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHTS_H
#define DEVICE_LIGHTS_H

#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightUtils.h"
#include "Device/includes/MISBSDFRayReuse.h"
#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/DI/FinalShading.h"
#include "Device/includes/RIS/RIS.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/SanityCheck.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_no_MIS(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    if (!MaterialUtils::can_do_light_sampling(ray_payload.material))
        return ColorRGB32F(0.0f);

    LightSampleInformation light_sample = sample_one_emissive_triangle(render_data, 
        closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, 
        closest_hit_info.primitive_index, ray_payload,
        random_number_generator);
    if (light_sample.area_measure_pdf <= 0.0f)
        // Can happen for very small triangles
        return ColorRGB32F(0.0f);

    float3 shadow_ray_origin = closest_hit_info.inter_point;
    float3 shadow_ray_direction = light_sample.point_on_light - shadow_ray_origin;
    float distance_to_light = hippt::length(shadow_ray_direction);
    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = shadow_ray_origin;
    shadow_ray.direction = shadow_ray_direction_normalized;

    ColorRGB32F light_source_radiance;
    // abs() here to allow backfacing light sources
    float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);
    if (dot_light_source > 0.0f)
    {
        NEEPlusPlusContext nee_plus_plus_context;
        nee_plus_plus_context.point_on_light = light_sample.point_on_light;
        nee_plus_plus_context.shaded_point = shadow_ray_origin;
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
                }
            }
        }
    }

    return light_source_radiance;
}

HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_bsdf(const HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, MISBSDFRayReuse& mis_ray_reuse)
{
    float bsdf_sample_pdf;
    float3 sampled_bsdf_direction;
    BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

    BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

    bool intersection_found = false;
    BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
    ColorRGB32F bsdf_radiance = ColorRGB32F(0.0f);
    if (bsdf_sample_pdf > 0.0f)
    {
        hiprtRay new_ray;
        new_ray.origin = closest_hit_info.inter_point;
        new_ray.direction = sampled_bsdf_direction;

        intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index, ray_payload.bounce, random_number_generator);

        // Checking that we did hit something and if we hit something,
        // it needs to be emissive
        if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black() && compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction) > 0.0f)
        {
            float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_bsdf_direction));
            bsdf_radiance = bsdf_color * cosine_term * shadow_light_ray_hit_info.hit_emission / bsdf_sample_pdf;

            // Just a CPU-only sanity check
            sanity_check</* CPUOnly */ true>(render_data, bsdf_radiance, 0, 0);
        }
    }

    // Note that if no intersection was found, the point computed here will
    // be garbage but we don't really care because if no intersection was found,
    // we're never going to read the intersection point anyway.
    // So it doesn't matter if it's garbage
    float3 bsdf_ray_inter_point = closest_hit_info.inter_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;
    mis_ray_reuse.fill(shadow_light_ray_hit_info, bsdf_ray_inter_point, sampled_bsdf_direction, bsdf_color, bsdf_sample_pdf, intersection_found ? RayState::BOUNCE : RayState::MISSED, incident_light_info);

    return bsdf_radiance;
}

HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_MIS(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, MISBSDFRayReuse& mis_ray_reuse)
{
    ColorRGB32F light_source_radiance_mis;

    if (MaterialUtils::can_do_light_sampling(ray_payload.material))
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle(render_data,
            closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, 
            closest_hit_info.primitive_index, ray_payload,
            random_number_generator);

        // Can happen for very small triangles that the PDF of the sampled triangle couldn't be computed
        if (light_sample.area_measure_pdf > 0.0f)
        {
            float3 shadow_ray_direction = light_sample.point_on_light - closest_hit_info.inter_point;
            float distance_to_light = hippt::length(shadow_ray_direction);
            float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

            hiprtRay shadow_ray;
            shadow_ray.origin = closest_hit_info.inter_point;
            shadow_ray.direction = shadow_ray_direction_normalized;

            NEEPlusPlusContext nee_plus_plus_context;
            nee_plus_plus_context.point_on_light = light_sample.point_on_light;
            nee_plus_plus_context.shaded_point = shadow_ray.origin;
            bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator, ray_payload.bounce);

            if (!in_shadow)
            {
                float bsdf_pdf;
                BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
                BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
                ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

                if (bsdf_pdf > 0.0f)
                {
                    float cos_theta_at_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

                    // Preventing division by 0 in the conversion to solid angle here
                    if (cos_theta_at_light_source > 1.0e-5f)
                    {
                        float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, cos_theta_at_light_source);

                        float light_pdf_for_mis = light_sample_pdf_for_MIS_solid_angle_measure(render_data,
                            light_sample_solid_angle_pdf,
                            light_sample.light_area, light_sample.emission, light_sample.light_source_normal,
                            distance_to_light, shadow_ray_direction_normalized,
                            
                            ray_payload.material.roughness, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, closest_hit_info.primitive_index, light_sample.point_on_light, random_number_generator);
                        float mis_weight = balance_heuristic(light_pdf_for_mis, bsdf_pdf);

                        float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));
                        light_source_radiance_mis = bsdf_color * cosine_term * light_sample.emission * mis_weight / light_sample_solid_angle_pdf / nee_plus_plus_context.unoccluded_probability;

                        // Just a CPU-only sanity check
                        sanity_check</* CPUOnly */ true>(render_data, light_source_radiance_mis, 0, 0);
                    }
                }
            }
        }
    }

    float bsdf_sample_pdf;
    float3 sampled_bsdf_direction;
    float3 bsdf_shadow_ray_origin = closest_hit_info.inter_point;
    BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
    ColorRGB32F bsdf_radiance_mis;

    unsigned int previous_seed = random_number_generator.m_state.seed;

    random_number_generator.m_state.seed = previous_seed;
    BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

    bool intersection_found = false;
    BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
    if (bsdf_sample_pdf > 0.0f)
    {
        hiprtRay new_ray;
        new_ray.origin = bsdf_shadow_ray_origin;
        new_ray.direction = sampled_bsdf_direction;

        intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index, ray_payload.bounce, random_number_generator);

        // Checking that we did hit something and if we hit something,
        // it needs to be emissive
        //
        // We're also checking if the light is backfacing maybe with compute_cosine_term()
        if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black() && compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction) > 0.0f)
        {
            float light_pdf_solid_angle = pdf_of_emissive_triangle_hit_solid_angle(render_data, shadow_light_ray_hit_info, sampled_bsdf_direction);
            float light_pdf_for_MIS = light_sample_pdf_for_MIS_solid_angle_measure(render_data, light_pdf_solid_angle, triangle_area(render_data, shadow_light_ray_hit_info.hit_prim_index), 
                shadow_light_ray_hit_info.hit_emission, shadow_light_ray_hit_info.hit_geometric_normal, shadow_light_ray_hit_info.hit_distance, 
                sampled_bsdf_direction,
                
                ray_payload.material.roughness, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, closest_hit_info.primitive_index, shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction + closest_hit_info.inter_point, random_number_generator);
            float mis_weight = balance_heuristic(bsdf_sample_pdf, light_pdf_for_MIS);

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
            bsdf_radiance_mis = bsdf_color * cosine_term * shadow_light_ray_hit_info.hit_emission * mis_weight / bsdf_sample_pdf;

            // Just a CPU-only sanity check
            sanity_check</* CPUOnly */ true>(render_data, bsdf_radiance_mis, 0, 0);
        }
    }

    // Note that if no intersection was found, the point computed here will
    // be garbage but we don't really care because if no intersection was found,
    // we're never going to read the intersection point anyway.
    // So it doesn't matter if it's garbage
    float3 bsdf_ray_inter_point = closest_hit_info.inter_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;
    mis_ray_reuse.fill(shadow_light_ray_hit_info, bsdf_ray_inter_point, sampled_bsdf_direction, bsdf_color, bsdf_sample_pdf, intersection_found ? RayState::BOUNCE : RayState::MISSED, incident_light_info);

    return light_source_radiance_mis + bsdf_radiance_mis;
}

HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_multiple_emissive_geometry(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, MISBSDFRayReuse& mis_ray_reuse)
{
    ColorRGB32F direct_light_contribution;

    // Any of these light sampling strategy support sampling multiple lights
    // per each shading point, effectively "amortizing" camera and bounce rays
    for (int i = 0; i < DirectLightSamplingNEESampleCount; i++)
    {
#if DirectLightSamplingStrategy == LSS_ONE_LIGHT
        direct_light_contribution += sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_BSDF
        direct_light_contribution += sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#elif DirectLightSamplingStrategy == LSS_MIS_LIGHT_BSDF
        direct_light_contribution += sample_one_light_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#elif DirectLightSamplingStrategy == LSS_RIS_BSDF_AND_LIGHT
        direct_light_contribution += sample_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#endif
    }

    return direct_light_contribution / DirectLightSamplingNEESampleCount;
}

HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_ReSTIR_DI(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, 
    const float3& view_direction, 
    Xorshift32Generator& random_number_generator, int2 pixel_coords, MISBSDFRayReuse& mis_ray_reuse)
{
    // ReSTIR DI doesn't support explicitely looping to sample
    // multiple lights per shading point so that's why we don't
    // have a loop for it

    ColorRGB32F direct_light_contribution;
    if (ray_payload.bounce == 0)
        // Can only do ReSTIR DI on the first bounce
        direct_light_contribution = sample_light_ReSTIR_DI(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, pixel_coords);
    else
    {
        // ReSTIR DI isn't used for the secondary/tertiary/... bounces
        // so there we can take multiple light samples per path vertex
        for (int i = 0; i < DirectLightSamplingNEESampleCount; i++)
        {
#if ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
            direct_light_contribution += sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_BSDF
            direct_light_contribution += sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF
            direct_light_contribution += sample_one_light_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT
            direct_light_contribution += sample_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#endif
        }

        direct_light_contribution /= DirectLightSamplingNEESampleCount;
    }

    return direct_light_contribution;
}

/**
 * Importance sample lights in the scene with NEE
 * 
 * Just a random note for myself and maybe future readers
 * that are wondering the same:
 * 
 * In the case where we shot a ray (camera ray or indirect bounce ray, doesn't matter)
 * and we hit an emissive material, we should still estimate NEE at that point. i.e. we
 * should also do NEE when standing on emissive materials because emissive materials can
 * reflect light just fine (unless they are blackbodies). 
 * 
 * Consider a glowing light bulb for example: this is just metal so hot that it glows
 * but because this is metal, it also reflects light.
 * 
 * I think the better morale to remember is that the material being emissive doesn't matter at
 * all. As long as the material itself reflects light, then we should do NEE.
 */
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F sample_emissive_geometry(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, 
    const float3& view_direction, 
    Xorshift32Generator& random_number_generator, int2 pixel_coords, MISBSDFRayReuse& mis_ray_reuse)
{
    if (render_data.buffers.emissive_triangles_count == 0 
        && !(render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP && DirectLightSamplingStrategy == LSS_RESTIR_DI))
        // No emissive geometry in the scene to sample
        // And we're not sampling the envmap with ReSTIR DI which means
        // that we're not sampling anything so return black
        return ColorRGB32F(0.0f);

    if (render_data.bsdfs_data.white_furnace_mode && render_data.bsdfs_data.white_furnace_mode_turn_off_emissives)
        return ColorRGB32F(0.0f);

    ColorRGB32F material_self_textured_emission;
    if (ray_payload.material.emissive_texture_used)
        // If the material is using an emissive texture, we will add its emission to the NEE estimation
        // because we're not importance sampling emissive textures so we're doing it the brute force
        // way for now (there are some things about light warping I think to properly sample emissive
        // textures but haven't read too much of that)
        material_self_textured_emission = ray_payload.material.emission;

    ColorRGB32F direct_light_contribution;
#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING
    direct_light_contribution = ColorRGB32F(0.0f);
#else // A light sampling strategy is used

#if DirectLightSamplingStrategy != LSS_RESTIR_DI
    // A light sampling strategy that is not ReSTIR DI
    // meaning that we can sample more than 1 light per
    // path vertex
    direct_light_contribution = sample_multiple_emissive_geometry(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);

#elif DirectLightSamplingStrategy == LSS_RESTIR_DI
    direct_light_contribution = sample_one_light_ReSTIR_DI(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, pixel_coords, mis_ray_reuse);
#endif
#endif

    return direct_light_contribution + material_self_textured_emission;
}

HIPRT_DEVICE ColorRGB32F clamp_direct_lighting_estimation(ColorRGB32F direct_lighting_contribution, float direct_contribution_clamp, int bounce)
{
    return clamp_light_contribution(direct_lighting_contribution, direct_contribution_clamp, bounce > 0);

}

/**
 * The x & y parameters are only used if using ReSTIR DI (they are for fetching the ReSTIR DI reservoir).
 * They can be ignored if not using ReSTIR DI
 */
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting(HIPRTRenderData& render_data, RayPayload& ray_payload, ColorRGB32F custom_ray_throughput, HitInfo& closest_hit_info,
    float3 view_direction,
    int x, int y,
    MISBSDFRayReuse& mis_reuse, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F total_direct_lighting;

    ColorRGB32F emissive_geometry_direct_contribution = sample_emissive_geometry(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, make_int2(x, y), mis_reuse);
    ColorRGB32F envmap_direct_contribution = sample_environment_map(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, mis_reuse);

    // Clamping direct lighting
    emissive_geometry_direct_contribution = clamp_light_contribution(emissive_geometry_direct_contribution, render_data.render_settings.direct_contribution_clamp, ray_payload.bounce == 0);
    envmap_direct_contribution = clamp_light_contribution(envmap_direct_contribution, render_data.render_settings.envmap_contribution_clamp, ray_payload.bounce == 0);

#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING // No direct light sampling
    ColorRGB32F hit_emission = ray_payload.material.emission;
    hit_emission = clamp_light_contribution(hit_emission, render_data.render_settings.indirect_contribution_clamp, ray_payload.bounce > 0);

    total_direct_lighting += hit_emission * custom_ray_throughput;
#else
    if (ray_payload.bounce == 0)
        // If we do have emissive geometry sampling, we only want to take
        // it into account on the first bounce, otherwise we would be
        // accounting for direct light sampling twice (bounce on emissive
        // geometry + direct light sampling). Otherwise, we don't check for bounce == 0
        total_direct_lighting += ray_payload.material.emission;

    // Clamped indirect lighting 
    ColorRGB32F direct_lighting_contribution = (emissive_geometry_direct_contribution + envmap_direct_contribution) * custom_ray_throughput;

    total_direct_lighting += direct_lighting_contribution;
#endif

    return total_direct_lighting;
}

/**
 * The x & y parameters are only used if using ReSTIR DI (they are for fetching the ReSTIR DI reservoir).
 * They can be ignored if not using ReSTIR DI
 */
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting_no_clamping(HIPRTRenderData& render_data, RayPayload& ray_payload, ColorRGB32F custom_ray_throughput, HitInfo& closest_hit_info,
    float3 view_direction,
    int x, int y,
    MISBSDFRayReuse& mis_reuse, Xorshift32Generator& random_number_generator)
{
    return estimate_direct_lighting(render_data, ray_payload, custom_ray_throughput, closest_hit_info, view_direction, x, y, mis_reuse, random_number_generator);
}

/**
 * The x & y parameters are only used if using ReSTIR DI (they are for fetching the ReSTIR DI reservoir).
 * They can be ignored if not using ReSTIR DI
 */
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting(HIPRTRenderData& render_data, RayPayload& ray_payload, HitInfo& closest_hit_info, 
    float3 view_direction,
    int x, int y,
    MISBSDFRayReuse& mis_reuse, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F unclamped_direct_lighting = estimate_direct_lighting(render_data, ray_payload, ray_payload.throughput, closest_hit_info, view_direction, x, y, mis_reuse, random_number_generator);

    return clamp_direct_lighting_estimation(unclamped_direct_lighting, render_data.render_settings.indirect_contribution_clamp, ray_payload.bounce);
}

#endif
