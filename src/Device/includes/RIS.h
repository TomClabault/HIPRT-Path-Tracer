/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RIS_H
#define DEVICE_RIS_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB evaluate_reservoir_sample(const HIPRTRenderData& render_data, const RendererMaterial& material, const float3& shading_point, const float3& shading_normal, const float3& view_direction, const Reservoir& reservoir)
{
    float3 evaluated_point = shading_point + shading_normal * 1.0e-4f;

    // Getting the sample
    if (reservoir.weight_sum == 0.0f)
        return ColorRGB(0.0f);

    float3 sample_direction = reservoir.sample.point_on_light_source - evaluated_point;
    float distance_to_light = hippt::length(sample_direction);
    float3 shadow_ray_direction_normalized = sample_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = evaluated_point;
    shadow_ray.direction = shadow_ray_direction_normalized;

    ColorRGB final_color;
    bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);
    if (!in_shadow)
    {
        float bsdf_pdf;
        RayVolumeState trash_volume_state;
        ColorRGB bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, shading_normal, shadow_ray.direction, bsdf_pdf);
        float cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(shading_normal, shadow_ray_direction_normalized));
        if (cosine_at_evaluated_point > 0.0f)
            final_color = bsdf_color * reservoir.UCW * reservoir.sample.emission * cosine_at_evaluated_point;// / distance_to_light / distance_to_light * hippt::abs(hippt::dot(reservoir.sample.light_source_normal, -shadow_ray_direction_normalized));
    }

    return final_color;
}

/**
 * Samples the lights in the scene and returns a out_reservoir that has "seen" (in the sense of Weighted Reservoir Sampling) all the samples
 */
HIPRT_HOST_DEVICE HIPRT_INLINE Reservoir sample_lights_RIS_reservoir(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    Reservoir out_reservoir;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;

    // Sampling candidates with weighted reservoir sampling
    for (int i = 0; i < render_data.render_settings.ris_number_of_light_candidates; i++)
    {
        float light_sample_pdf;
        float distance_to_light;
        float cosine_at_light_source;
        float cosine_at_evaluated_point;
        RayVolumeState trash_ray_volume_state;
        LightSourceInformation light_source_info;

        ColorRGB bsdf_color;
        float target_function = 0.0f;
        float candidate_weight = 0.0f;
        float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);

        // It can happen that the light PDF returned by the emissive triangle
        // sampling function is 0 because of emissive triangles that are so
        // small that we cannot compute their normal and their area (the cross
        // product of their edges gives a quasi-null vector --> length of 0.0f --> area of 0)
        if (light_sample_pdf > 0.0f)
        {
            float bsdf_pdf;
            float3 to_light_direction;

            to_light_direction = random_light_point - evaluated_point;
            distance_to_light = hippt::length(to_light_direction);
            to_light_direction = to_light_direction / distance_to_light; // Normalization
            cosine_at_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -to_light_direction));
            cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, to_light_direction));
            if (cosine_at_evaluated_point > 0.0f)
            {
                bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_ray_volume_state, view_direction, closest_hit_info.shading_normal, to_light_direction, bsdf_pdf);
                // Converting the PDF from area measure to solid angle measure requires dividing by
                // cos(theta) / dist^2. Dividing by that factor is equal to multiplying by the inverse
                // which is what we're doing here
                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= cosine_at_light_source;
                light_sample_pdf /= render_data.buffers.emissive_triangles_count;

                float geometry_term = 1.0f / distance_to_light / distance_to_light;

                target_function = (bsdf_color * light_source_info.emission * cosine_at_evaluated_point * geometry_term).luminance();

#if RISUseVisiblityTargetFunction == RIS_USE_VISIBILITY_TRUE
                // Adding visibility to the target function
                hiprtRay shadow_ray;
                shadow_ray.origin = evaluated_point;
                shadow_ray.direction = to_light_direction;

                bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

                target_function *= visible;
#endif

                float mis_weight = balance_heuristic(light_sample_pdf, render_data.render_settings.ris_number_of_light_candidates, bsdf_pdf, render_data.render_settings.ris_number_of_bsdf_candidates);
                candidate_weight = mis_weight * target_function / light_sample_pdf;
            }
        }

        ReservoirSample sample;
        sample.point_on_light_source = random_light_point;
        sample.emission = light_source_info.emission;
        sample.target_function = target_function;
        sample.light_source_normal = light_source_info.light_source_normal;

        out_reservoir.add_one_candidate(sample, candidate_weight, random_number_generator);
    }

    for (int i = 0; i < render_data.render_settings.ris_number_of_bsdf_candidates; i++)
    {
        float bsdf_sample_pdf = 0.0f;
        float target_function = 0.0f;
        float candidate_weight = 0.0f;
        float cosine_at_evaluated_point = 0.0f;
        float cosine_light_source = 0.0f;
        float3 sampled_direction;
        ColorRGB bsdf_color;
        RayVolumeState trash_ray_volume_state;

        bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, material, trash_ray_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction, bsdf_sample_pdf, random_number_generator);
        cosine_at_evaluated_point = hippt::dot(closest_hit_info.shading_normal, sampled_direction);

        ReservoirSample sample;
        if (bsdf_sample_pdf > 0.0f && cosine_at_evaluated_point > 0.0f)
        {
            hiprtRay bsdf_ray;
            bsdf_ray.origin = evaluated_point;
            bsdf_ray.direction = sampled_direction;

            HitInfo bsdf_ray_hit_info;
            RayPayload ray_payload;
            bool hit_found = trace_ray(render_data, bsdf_ray, ray_payload, bsdf_ray_hit_info);
            if (hit_found && ray_payload.material.is_emissive())
            {
                // If we intersected an emissive material, compute the weight. 
                // Otherwise, the weight is 0 because of the emision being 0 so we just don't compute it

                cosine_light_source = hippt::abs(hippt::dot(bsdf_ray_hit_info.shading_normal, -sampled_direction));

                float geometry_term = cosine_at_evaluated_point * cosine_light_source / bsdf_ray_hit_info.t / bsdf_ray_hit_info.t;
                target_function = (bsdf_color * ray_payload.material.emission * cosine_at_evaluated_point * geometry_term).luminance();

                float light_area = triangle_area(render_data, bsdf_ray_hit_info.primitive_index);
                float light_pdf = bsdf_ray_hit_info.t* bsdf_ray_hit_info.t / cosine_light_source;
                light_pdf /= light_area;
                light_pdf /= render_data.buffers.emissive_triangles_count;

                float mis_weight = balance_heuristic(bsdf_sample_pdf, render_data.render_settings.ris_number_of_bsdf_candidates, light_pdf, render_data.render_settings.ris_number_of_light_candidates);
                candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

                sample.emission = ray_payload.material.emission;
                sample.point_on_light_source = bsdf_ray_hit_info.inter_point;
                sample.light_source_normal = bsdf_ray_hit_info.shading_normal;
                sample.target_function = target_function;
            }
        }

        out_reservoir.add_one_candidate(sample, candidate_weight, random_number_generator);
    }

    out_reservoir.end();
    return out_reservoir;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_lights_RIS(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    Reservoir reservoir = sample_lights_RIS_reservoir(render_data, material, closest_hit_info, view_direction, random_number_generator);

    return evaluate_reservoir_sample(render_data, material, closest_hit_info.inter_point, closest_hit_info.shading_normal, view_direction, reservoir);
}

#endif
