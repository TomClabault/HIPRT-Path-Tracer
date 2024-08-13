/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_UTILS_H
#define DEVICE_LIGHT_UTILS_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float3 sample_one_emissive_triangle(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    int random_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    //random_index = 1;
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
    // TODO remove hardcoded
    //random_point_on_triangle = (vertex_A + vertex_B + vertex_C) / 3;

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

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F clamp_light_contribution(ColorRGB32F light_contribution, float clamp_max_value, bool clamp_condition)
{
    if (!light_contribution.has_NaN() && clamp_max_value > 0.0f && clamp_condition)
        // We don't want to clamp NaNs because that's UB (kind of) and the NaNs get
        // immediately clamped to 'clamp_max_value' in my experience
        light_contribution.clamp(0.0f, clamp_max_value);

    return light_contribution;
}

#endif
