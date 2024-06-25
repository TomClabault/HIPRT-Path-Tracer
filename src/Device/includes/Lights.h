/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHTS_H
#define DEVICE_LIGHTS_H

#include "HostDeviceCommon/KernelOptions.h"
#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float3 sample_one_emissive_triangle(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    int random_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = render_data.buffers.emissive_triangles_indices[random_index];

    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 2]];

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;

    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;

    float3 normal = hippt::cross(AB, AC);
    float length_normal = hippt::length(normal);
    if (length_normal <= 1.0e-6f)
    {
        // Can happen with very small triangles
        pdf = 0.0f;

        return make_float3(0, 0, 0);
    }


    light_info.emissive_triangle_index = triangle_index;
    light_info.light_source_normal = normal / length_normal; // Normalization
    light_info.light_area = length_normal * 0.5f;
    light_info.emission = render_data.buffers.materials_buffer[render_data.buffers.material_indices[triangle_index]].emission;

    pdf = 1.0f / light_info.light_area;

    return random_point_on_triangle;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float triangle_area(const HIPRTRenderData& render_data, int triangle_index)
{
    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 2]];

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;

    return hippt::length(hippt::cross(AB, AC)) / 2.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_one_light_no_MIS(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    ColorRGB light_source_radiance;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
    if (!(light_sample_pdf > 0.0f))
        // Can happen for very small triangles
        return ColorRGB(0.0f);

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
            // Conversion to solid angle from surface area measure
            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;
            light_sample_pdf /= render_data.buffers.emissive_triangles_count;

            float brdf_pdf;
            RayVolumeState trash_volume_state;
            ColorRGB bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, brdf_pdf);
            if (brdf_pdf != 0.0f)
            {
                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance = light_source_info.emission * cosine_term * bsdf_color / light_sample_pdf;
            }
        }
    }

    return light_source_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_one_light_bsdf(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    ColorRGB bsdf_radiance;

    float3 sampled_brdf_direction;
    float direction_pdf;
    RayVolumeState trash_volume_state;
    ColorRGB bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
    if (direction_pdf > 0.0f)
    {
        hiprtRay new_ray;
        new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
        new_ray.direction = sampled_brdf_direction;

        HitInfo new_ray_hit_info;
        RayPayload bsdf_ray_payload;
        bool inter_found = trace_ray(render_data, new_ray, bsdf_ray_payload, new_ray_hit_info);

        // Checking that we did hit something and if we hit something,
        // it needs to be the light that we're currently sampling
        if (inter_found && bsdf_ray_payload.material.is_emissive())
        {
            float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            bsdf_radiance = bsdf_color * cosine_term * bsdf_ray_payload.material.emission / direction_pdf;
        }
    }

    return bsdf_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_one_light_MIS(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float light_sample_pdf;
    ColorRGB light_source_radiance_mis;
    LightSourceInformation light_source_info;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
    if (light_sample_pdf <= 0.0f)
        // Can happen for very small triangles
        return ColorRGB(0.0f);

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
            // Conversion to solid angle from surface area measure
            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;

            float bsdf_pdf;
            RayVolumeState trash_volume_state;
            ColorRGB bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, bsdf_pdf);
            if (bsdf_pdf != 0.0f)
            {
                float mis_weight = power_heuristic(light_sample_pdf, bsdf_pdf);

                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance_mis = bsdf_color * cosine_term * light_source_info.emission * mis_weight / light_sample_pdf;
            }
        }
    }

    ColorRGB bsdf_radiance_mis;

    float3 sampled_brdf_direction;
    float direction_pdf;
    RayVolumeState trash_volume_state;
    ColorRGB bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
    if (direction_pdf > 0)
    {
        hiprtRay new_ray;
        new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
        new_ray.direction = sampled_brdf_direction;

        HitInfo new_ray_hit_info;
        RayPayload trash_payload;
        bool inter_found = trace_ray(render_data, new_ray, trash_payload, new_ray_hit_info);

        // Checking that we did hit something and if we hit something,
        // it needs to be the light that we're currently sampling
        if (inter_found && new_ray_hit_info.primitive_index == light_source_info.emissive_triangle_index)
        {
            // abs() here to allow double sided emissive geometry.
            // Without abs() here:
            //  - We could be hitting the back of an emissive triangle 
            //  --> triangle normal not facing the same way 
            //  --> cos_angle negative
            float cos_angle_light = hippt::abs(hippt::dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction));

            float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
            float light_pdf = distance_squared / (light_source_info.light_area * cos_angle_light);
            float mis_weight = power_heuristic(direction_pdf, light_pdf);

            float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            bsdf_radiance_mis = bsdf_color * cosine_term * light_source_info.emission * mis_weight / direction_pdf;
        }
    }

    // Because we're sampling only 1 light out of all the lights of the
    // scene, the probability of having chosen that light is: 1 / numberOfLights
    // This must be factored in the PDF of sampling that light which means that we must
    // divide by 1 / numberOfLights => multiply by numberOfLights
    return (light_source_radiance_mis + bsdf_radiance_mis) * render_data.buffers.emissive_triangles_count;
}

struct ReservoirSample
{
    // Light sample
    float3 point_on_light_source = { 0, 0, 0 };
    ColorRGB emission = { 0.0f, 0.0f, 0.0f };
};

struct Reservoir
{
    HIPRT_HOST_DEVICE void update(ReservoirSample new_sample, float weight, Xorshift32Generator& random_number_generator)
    {
        M++;
        weight_sum += weight;

        if (random_number_generator() < weight / weight_sum)
            sample = new_sample;
    }

    /*HIPRT_HOST_DEVICE float get_W()
    {
        return 1.0f / sample.target_function_weight * weight_sum;
    }*/

    HIPRT_HOST_DEVICE ReservoirSample get_sample()
    {
        return sample;
    }

    unsigned int M = 0;
    float weight_sum = 0.0f;

    ReservoirSample sample;
};

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_bsdf_and_lights_RIS(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;

    // Sampling candidates with weighted reservoir sampling
    Reservoir reservoir;
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
        if (light_sample_pdf > 0.0f)
        {
            // It can happen that the light PDF returned by the emissive triangle
            // sampling function is 0 because of emissive triangles that are so
            // small that we cannot compute their normal and their area (the cross
            // product of their edges gives a quasi-null vector --> length of 0.0f --> area of 0)
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

                float geometry_term = 1.0f / (distance_to_light * distance_to_light) * cosine_at_light_source * cosine_at_evaluated_point;
                target_function = bsdf_color.length() * light_source_info.emission.length() * cosine_at_evaluated_point;

#if RISUseVisiblityTargetFunction == RIS_USE_VISIBILITY_TRUE
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

        reservoir.update(sample, candidate_weight, random_number_generator);
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

        ReservoirSample new_sample;
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

                //float geometry_term = 1.0f / (bsdf_ray_hit_info.t * bsdf_ray_hit_info.t) * cosine_at_evaluated_point * cosine_light_source;
                target_function = bsdf_color.length() * ray_payload.material.emission.length() * cosine_at_evaluated_point;

                float light_area = triangle_area(render_data, bsdf_ray_hit_info.primitive_index);
                float light_pdf = bsdf_ray_hit_info.t * bsdf_ray_hit_info.t / cosine_light_source;
                light_pdf /= light_area;
                light_pdf /= render_data.buffers.emissive_triangles_count;

                float mis_weight = balance_heuristic(bsdf_sample_pdf, render_data.render_settings.ris_number_of_bsdf_candidates, light_pdf, render_data.render_settings.ris_number_of_light_candidates);
                candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

                new_sample.emission = ray_payload.material.emission;
                new_sample.point_on_light_source = bsdf_ray_hit_info.inter_point;
            }
        }

        reservoir.update(new_sample, candidate_weight, random_number_generator);
    }

    ColorRGB final_color;

    // Getting the sample
    if (reservoir.weight_sum == 0.0f)
        return ColorRGB(0.0f);

    ReservoirSample sample = reservoir.get_sample();

    float3 shadow_ray_direction = sample.point_on_light_source - evaluated_point;
    float distance_to_light = hippt::length(shadow_ray_direction);
    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = evaluated_point;
    shadow_ray.direction = shadow_ray_direction_normalized;

    bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);
    if (!in_shadow)
    {
        float brdf_pdf;
        float cosine_at_evaluated_point;
        ColorRGB bsdf_color;
        RayVolumeState trash_volume_state;

        bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, brdf_pdf);
        cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, shadow_ray_direction_normalized));
        if (cosine_at_evaluated_point > 0.0f)
        {
            // Visibility of the target function is already included because we can only be here it the light wasn't occluded
            float target_function = bsdf_color.length() * sample.emission.length() * cosine_at_evaluated_point;
            float UCW = 1.0f / target_function * reservoir.weight_sum;

            final_color = bsdf_color * UCW * sample.emission * cosine_at_evaluated_point;
        }
    }

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_one_light(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        // No emmisive geometry in the scene to sample
        return ColorRGB(0.0f);

    if (material.emission.r != 0.0f || material.emission.g != 0.0f || material.emission.b != 0.0f)
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return ColorRGB(0.0f);

    if (hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0.0f)
        // We're not direct sampling if we're inside a surface
        // 
        // We're using the geometric normal here because using the shading normal could lead
        // to false positive because of the black fringes when using smooth normals / normal mapping
        // + microfacet BRDFs
        return ColorRGB(0.0f);

#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING
    return ColorRGB(0.0f);
#elif DirectLightSamplingStrategy == LSS_UNIFORM_ONE_LIGHT
    return sample_one_light_no_MIS(render_data, material, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_BSDF
    return sample_one_light_bsdf(render_data, material, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_MIS_LIGHT_BSDF
    return sample_one_light_MIS(render_data, material, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_RIS_BSDF_AND_LIGHT
    return sample_bsdf_and_lights_RIS(render_data, material, closest_hit_info, view_direction, random_number_generator);
#endif
}

#endif
