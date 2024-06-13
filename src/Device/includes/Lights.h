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

    light_info.emissive_triangle_index = triangle_index;
    light_info.light_source_normal = normal / length_normal; // Normalization
    light_info.light_area = length_normal * 0.5f;
    light_info.emission = render_data.buffers.materials_buffer[render_data.buffers.material_indices[triangle_index]].emission;

    pdf = 1.0f / light_info.light_area;

    return random_point_on_triangle;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_one_light_no_MIS(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    ColorRGB light_source_radiance_mis;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);

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
            ColorRGB brdf = brdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, brdf_pdf);
            if (brdf_pdf != 0.0f)
            {
                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance_mis = light_source_info.emission * cosine_term * brdf / light_sample_pdf;
            }
        }
    }

    return light_source_radiance_mis;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_one_light_MIS(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float light_sample_pdf;
    ColorRGB light_source_radiance_mis;
    LightSourceInformation light_source_info;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);

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

            float brdf_pdf;
            RayVolumeState trash_volume_state;
            ColorRGB brdf = brdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, brdf_pdf);
            if (brdf_pdf != 0.0f)
            {
                float mis_weight = power_heuristic(light_sample_pdf, brdf_pdf);

                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance_mis = light_source_info.emission * cosine_term * brdf * mis_weight / light_sample_pdf;
            }
        }
    }

    ColorRGB brdf_radiance_mis;

    float3 sampled_brdf_direction;
    float direction_pdf;
    RayVolumeState trash_volume_state;
    ColorRGB brdf = brdf_dispatcher_sample(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
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
            float cos_angle = hippt::abs(hippt::dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction));

            float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
            float light_pdf = distance_squared / (light_source_info.light_area * cos_angle);
            float mis_weight = power_heuristic(direction_pdf, light_pdf);

            float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            brdf_radiance_mis = brdf * cosine_term * light_source_info.emission * mis_weight / direction_pdf;
        }
    }

    // Because we're sampling only 1 light out of all the lights of the
    // scene, the probability of having chosen that light is: 1 / numberOfLights
    // This must be factored in the PDF of sampling that light which means that we must
    // divide by 1 / numberOfLights => multiply by numberOfLights
    return (light_source_radiance_mis + brdf_radiance_mis) * render_data.buffers.emissive_triangles_count;
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

#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING // No direct light sampling
    return ColorRGB(0.0f);
#elif DirectLightSamplingStrategy == LSS_ONE_RANDOM_LIGHT // No MIS
    return sample_one_light_no_MIS(render_data, material, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_ONE_RANDOM_LIGHT_MIS // MIS
    return sample_one_light_MIS(render_data, material, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_ONE_RANDOM_LIGHT_RIS // RIS
    return ColorRGB(0.0f); // Not implemented yet
#endif
}

#endif
