/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHTS_H
#define DEVICE_LIGHTS_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float3 sample_random_point_on_lights(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    int random_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = light_info.emissive_triangle_index = render_data.buffers.emissive_triangles_indices[random_index];

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
    light_info.light_source_normal = normal / length_normal; // Normalization
    float triangle_area = length_normal * 0.5f;
    float nb_emissive_triangles = render_data.buffers.emissive_triangles_count;

    pdf = 1.0f / (nb_emissive_triangles * triangle_area);

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

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_light_sources(const HIPRTRenderData& render_data, const RendererMaterial& material, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
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


    ColorRGB light_source_radiance_mis;
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    float3 random_light_point = sample_random_point_on_lights(render_data, random_number_generator, light_sample_pdf, light_source_info);

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
            const RendererMaterial& emissive_triangle_material = render_data.buffers.materials_buffer[render_data.buffers.material_indices[light_source_info.emissive_triangle_index]];

            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;

            float pdf;
            RayVolumeState trash_volume_state;
            ColorRGB brdf = brdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, pdf);
            if (pdf != 0.0f)
            {
                float mis_weight = power_heuristic(light_sample_pdf, pdf);

                ColorRGB Li = emissive_triangle_material.emission;
                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);

                light_source_radiance_mis = Li * cosine_term * brdf * mis_weight / light_sample_pdf;
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

        if (inter_found)
        {
            // abs() here to allow double sided emissive geometry.
            // Without abs() here:
            //  - We could be hitting the back of an emissive triangle 
            //  --> triangle normal not facing the same way 
            //  --> cos_angle negative
            float cos_angle = hippt::abs(hippt::dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction));
            int material_index = render_data.buffers.material_indices[new_ray_hit_info.primitive_index];
            RendererMaterial material = render_data.buffers.materials_buffer[material_index];

            if (material.is_emissive())
            {
                float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
                float light_area = triangle_area(render_data, new_ray_hit_info.primitive_index);

                float light_pdf = distance_squared / (light_area * cos_angle);

                float mis_weight = power_heuristic(direction_pdf, light_pdf);
                float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
                brdf_radiance_mis = brdf * cosine_term * material.emission * mis_weight / direction_pdf;
            }
        }
    }

    return light_source_radiance_mis + brdf_radiance_mis;
}

#endif
