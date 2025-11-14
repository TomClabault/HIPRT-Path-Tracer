/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H
 
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/TriangleSamplingSolidAngle.h"
#include "Device/includes/LightSampling/TriangleSamplingProjectedSolidAngle.h"
#include "Device/includes/TriangleLoadUtils.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

/**
 * Reference: [A Low-Distortion Map Between Triangle and Square, Heitz, 2019]
 *
 * Maps a point in a square to a point in an arbitrary triangle
 */
HIPRT_DEVICE float2 square_to_triangle(float& x, float& y)
{
    if (y > x)
    {
        x *= 0.5f;
        y -= x;
    }
    else
    {
        y *= 0.5f;
        x -= y;
    }

    return make_float2(x, y);
}

HIPRT_DEVICE float3 sample_point_on_triangle_uniform_area(float3 vertex_A, float3 edge_AB, float3 edge_AC, float triangle_area, Xorshift32Generator& rng, float& out_point_pdf)
{
    float rand_1 = rng();
    float rand_2 = rng();

#if TrianglePointSamplingUniformAreaStrategy == TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_TURK_1990
    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;
#elif TrianglePointSamplingUniformAreaStrategy == TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_HEITZ_2019
    float2 remapped = square_to_triangle(rand_1, rand_2);

    float u = remapped.x;
    float v = remapped.y;
#endif

    out_point_pdf = 1.0f / triangle_area;

    return vertex_A + edge_AB * u + edge_AC * v;
}

/**
 * Samples a point uniformly on the given triangle (given with the triangle index)
 *
 * Returns true if the sampling was successful, false otherwise (can fail if the triangle is way too small or degenerate)
 */
HIPRT_DEVICE bool sample_point_on_generic_triangle(const HIPRTRenderData& render_data, 
    float3 shading_point, float3 view_direction, float3 shading_normal,
    int global_triangle_index, Xorshift32Generator& rng,
    float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area, 
    float& out_point_pdf)
{
    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[global_triangle_index * 3 + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[global_triangle_index * 3 + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[global_triangle_index * 3 + 2]];

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;
    float3 normal = hippt::cross(AB, AC);

    float length_normal = hippt::length(normal);
    if (length_normal <= TriangleSamplingNormalLengthRejectionThreshold)
        return false;

    normal /= length_normal;

    out_sampled_triangle_normal = normal;
    out_triangle_area = 0.5f * length_normal;

#if TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA
    out_sample_point = sample_point_on_triangle_uniform_area(vertex_A, AB, AC, out_triangle_area, rng, out_point_pdf);
#elif TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE
    out_sample_point = sample_point_on_triangle_solid_angle_peters_2021(vertex_A, vertex_B, vertex_C, normal, shading_point, out_point_pdf, rng);
#elif TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE
    float solid_angle = triangle_solid_angle(vertex_A, vertex_B, vertex_C, shading_point);

    bool do_projected_solid_angle_sampling = solid_angle > render_data.render_settings.projected_solid_angle_sampling_threshold;
    if (do_projected_solid_angle_sampling)
        // If the triangle is large enough in solid angle, it may be worth it to compute the heavy projected solid angle
        // stuff
        out_sample_point = sample_point_on_triangle_projected_solid_angle_peters_2021(render_data,
            vertex_A, vertex_B, vertex_C, normal, 
            shading_point, view_direction, shading_normal, out_point_pdf, rng);
    else
        // Otherwise it's not worth it and we can use the cheap solid angle (not projected) sampling
        out_sample_point = sample_point_on_triangle_solid_angle_peters_2021(vertex_A, vertex_B, vertex_C, normal, shading_point, out_point_pdf, rng);
#endif

    return out_point_pdf != 0.0f;
}

/**
 * From a triangle index, samples uniformly a point on the triangle and fills a LightSampleInformation
 * structure with the information (normal, area, emission, ...) of the triangle
 *
 * The PDF field of the LightSampleInformation is only field with the probability of sampling the
 * point on the triangle. The rest of the PDF must be computed by the caller
 */
HIPRT_DEVICE LightSampleInformation sample_point_on_generic_triangle_and_fill_light_sample_information(const HIPRTRenderData& render_data, 
    float3 shading_point, float3 view_direction, float3 shading_normal, 
    int global_triangle_index, Xorshift32Generator& rng)
{
    LightSampleInformation light_sample;

    float sampled_point_pdf;
    float sampled_triangle_area;
    float3 sampled_triangle_normal;
    float3 random_point_on_triangle;
    unsigned int point_on_light_random_seed;
    if (!sample_point_on_generic_triangle(render_data,
        shading_point, view_direction, shading_normal,
        global_triangle_index, rng, 
        random_point_on_triangle, sampled_triangle_normal, sampled_triangle_area, sampled_point_pdf))
        return LightSampleInformation();

    light_sample.emissive_triangle_global_index = global_triangle_index;
    light_sample.light_source_normal = sampled_triangle_normal;
    light_sample.light_area = sampled_triangle_area;
    light_sample.emission = render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[global_triangle_index]);
    light_sample.point_on_light = random_point_on_triangle;
    light_sample.area_measure_pdf = sampled_point_pdf;

    return light_sample;
}

#endif
