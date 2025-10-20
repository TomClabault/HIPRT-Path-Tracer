/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H
 
#include "Device/includes/LightSampling/LightSampleInformation.h"
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

/**
 * Samples a point uniformly on the given triangle (given with the triangle index)
 *
 * Returns true if the sampling was successful, false otherwise (can fail if the triangle is way too small or degenerate)
 */
HIPRT_DEVICE bool sample_point_on_generic_triangle(int global_triangle_index, const float3* vertices_positions, const int* triangles_indices, Xorshift32Generator& rng,
    float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area, unsigned int& out_point_on_light_random_seed)
{
    float3 vertex_A = vertices_positions[triangles_indices[global_triangle_index * 3 + 0]];
    float3 vertex_B = vertices_positions[triangles_indices[global_triangle_index * 3 + 1]];
    float3 vertex_C = vertices_positions[triangles_indices[global_triangle_index * 3 + 2]];

    out_point_on_light_random_seed = rng.m_state.seed;
    float rand_1 = rng();
    float rand_2 = rng();

#if TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_TURK_1990
    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;
#elif TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_HEITZ_2019
    float2 remapped = square_to_triangle(rand_1, rand_2);

    float u = remapped.x;
    float v = remapped.y;
#endif

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;
    float3 normal = hippt::cross(AB, AC);

    float length_normal = hippt::length(normal);
    if (length_normal <= TriangleSamplingNormalLengthRejectionThreshold)
        return false;

    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;
    out_sample_point = random_point_on_triangle;
    out_sampled_triangle_normal = normal / length_normal;
    out_triangle_area = 0.5f * length_normal;

    return true;
}

HIPRT_DEVICE bool sample_point_on_generic_triangle(int global_triangle_index, const float3* vertices_positions, const int* triangles_indices, Xorshift32Generator& rng,
    float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area)
{
    unsigned int trash_random_seed;
    return sample_point_on_generic_triangle(global_triangle_index, vertices_positions, triangles_indices, rng, out_sample_point, out_sampled_triangle_normal, out_triangle_area, trash_random_seed);
}

/**
 * From a triangle index, samples uniformly a point on the triangle and fills a LightSampleInformation
 * structure with the information (normal, area, emission, ...) of the triangle
 *
 * The PDF field of the LightSampleInformation is only field with the probability of sampling the
 * point on the triangle. The rest of the PDF must be computed by the caller
 */
HIPRT_DEVICE LightSampleInformation sample_point_on_generic_triangle_and_fill_light_sample_information(const HIPRTRenderData& render_data, int global_triangle_index, Xorshift32Generator& rng)
{
    LightSampleInformation light_sample;

    float sampled_triangle_area;
    float3 sampled_triangle_normal;
    float3 random_point_on_triangle;
    unsigned int point_on_light_random_seed;
    if (!sample_point_on_generic_triangle(global_triangle_index, render_data.buffers.vertices_positions,
        render_data.buffers.triangles_indices, rng, random_point_on_triangle, sampled_triangle_normal, sampled_triangle_area, point_on_light_random_seed))
        return LightSampleInformation();

    light_sample.emissive_triangle_global_index = global_triangle_index;
    light_sample.light_source_normal = sampled_triangle_normal;
    light_sample.light_area = sampled_triangle_area;
    light_sample.emission = render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[global_triangle_index]);
    light_sample.point_on_light = random_point_on_triangle;
    light_sample.area_measure_pdf = 1.0f / light_sample.light_area;
#if DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
    // Only needed for ReGIR
    light_sample.sample_random_seed = point_on_light_random_seed;
#endif

    return light_sample;
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, unsigned int point_on_light_random_seed, unsigned int emissive_triangle_global_index, float3& out_triangle_normal, float& out_triangle_area)
{
    Xorshift32Generator rng(point_on_light_random_seed);

    float3 sampled_point;
    if (!sample_point_on_generic_triangle(emissive_triangle_global_index, render_data.buffers.vertices_positions, render_data.buffers.triangles_indices, rng,
        sampled_point, out_triangle_normal, out_triangle_area))
        return make_float3(-1.0e35f, -1.0e35f, -1.0e35f);

    return sampled_point;
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, unsigned int point_on_light_random_seed, unsigned int emissive_triangle_global_index, float3& out_triangle_normal)
{
    float trash_area;
    return reconstruct_sample_point_on_light(render_data, point_on_light_random_seed, emissive_triangle_global_index, out_triangle_normal, trash_area);
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, const ReGIRSample& sample, float3& out_triangle_normal, float& out_triangle_area)
{
    return reconstruct_sample_point_on_light(render_data, sample.point_on_light_random_seed, sample.emissive_triangle_global_index, out_triangle_normal, out_triangle_area);
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, const ReGIRSample& sample, float3& out_triangle_normal)
{
    return reconstruct_sample_point_on_light(render_data, sample.point_on_light_random_seed, sample.emissive_triangle_global_index, out_triangle_normal);
}

#endif
