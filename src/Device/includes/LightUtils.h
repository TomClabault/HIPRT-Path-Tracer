/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_UTILS_H
#define DEVICE_LIGHT_UTILS_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * The PDF is computed in area measure
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 uniform_sample_one_emissive_triangle(
    const int* triangles_indices, const int* emissive_triangles_indices, int emissive_triangle_count,
    const float3* vertices_positions,
    const int* material_indices, const DevicePackedTexturedMaterialSoA& materials_buffer,
    Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    //int random_index = 0;
    int random_index = random_number_generator.random_index(emissive_triangle_count);
    int triangle_index = emissive_triangles_indices[random_index];

    float3 vertex_A = vertices_positions[triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = vertices_positions[triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = vertices_positions[triangles_indices[triangle_index * 3 + 2]];

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();
    //float rand_1 = 0.258484f;
    //float rand_2 = 0.00124f;

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
    light_info.emission = materials_buffer.get_emission(material_indices[triangle_index]);

    pdf = 1.0f / light_info.light_area;
    pdf /= emissive_triangle_count;

    return random_point_on_triangle;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 uniform_sample_one_emissive_triangle(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    return uniform_sample_one_emissive_triangle(render_data.buffers.triangles_indices, render_data.buffers.emissive_triangles_indices, render_data.buffers.emissive_triangles_count,
        render_data.buffers.vertices_positions,
        render_data.buffers.material_indices, render_data.buffers.materials_buffer,
        random_number_generator, pdf, light_info);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 get_triangle_normal_not_normalized(const HIPRTRenderData& render_data, int triangle_index)
{
    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 2]];

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;

    return hippt::cross(AB, AC);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float triangle_area(const HIPRTRenderData& render_data, int triangle_index)
{
    float3 normal = get_triangle_normal_not_normalized(render_data, triangle_index);
    return hippt::length(normal) * 0.5f;
}

HIPRT_INLINE HIPRT_HOST_DEVICE float area_to_solid_angle_pdf(float area_pdf, float distance, float cos_theta)
{
    if (cos_theta < 1.0e-8f)
        return 0.0f;

    return area_pdf * hippt::square(distance) / cos_theta;
}

/**
 * 'clamp_condition' is an additional condition that needs to be met
 * for clamping to occur. If the additional condition is not met (the boolean
 * 'clamp_condition' is false, then the 'light_contribution' parameter is returned
 * untouched
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F clamp_light_contribution(ColorRGB32F light_contribution, float clamp_max_value, bool clamp_condition)
{
    if (!light_contribution.has_nan() && clamp_max_value > 0.0f && clamp_condition)
        // We don't want to clamp NaNs because that's UB (kind of) and the NaNs get
        // immediately clamped to 'clamp_max_value' in my experience
        //
        // Not clamping the negatives to 0 because
        // spectral rendering (for dispersion for example) may produce negative values
        // and we don't want to clamp those to 0
        light_contribution.clamp(-clamp_max_value, clamp_max_value);

    return light_contribution;
}

/**
 * Returns the PDF (area measure) of the light sampler for the given triangle_hit_info
 *
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit
 * 'hit_distance' is the distance to the intersection point on the hit triangle
 * 'ray_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, const ShadowLightRayHitInfo& light_hit_info)
{
    // Surface area PDF of hitting that point on that triangle in the scene
    float light_area = triangle_area(render_data, light_hit_info.hit_prim_index);
    float area_measure_pdf = 1.0f / light_area;
    area_measure_pdf /= render_data.buffers.emissive_triangles_count;

    return area_measure_pdf;
}

/**
 * Returns the PDF (solid angle measure) of the light sampler for the given triangle_hit_info
 * 
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit
 * 'hit_distance' is the distance to the intersection point on the hit triangle
 * 'ray_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, const ShadowLightRayHitInfo& light_hit_info, float3 ray_direction)
{
    // abs() here to allow backfacing lights
    // Without abs() here:
    //  - We could be hitting the back of an emissive triangle (think of quad light hanging in the air)
    //  --> triangle normal not facing the same way 
    //  --> cos_angle negative
    float cosine_light_source = hippt::abs(hippt::dot(light_hit_info.hit_shading_normal, ray_direction));

    float pdf_area_measure = pdf_of_emissive_triangle_hit_area_measure(render_data, light_hit_info);
    return area_to_solid_angle_pdf(pdf_area_measure, light_hit_info.hit_distance, cosine_light_source);
}

/**
 * Returns true if the given contribution satisfies the minimum light contribution
 * required for a light to be 
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool check_minimum_light_contribution(float minimum_contribution, const ColorRGB32F& contribution)
{
    if (minimum_contribution > 0.0f)
    {
        if (contribution.r < minimum_contribution
            && contribution.g < minimum_contribution
            && contribution.b < minimum_contribution)
            // The light doesn't contribute enough
            return false;
        else
            // The light contributes enough
            return true;
    }
    else
        // Minimum light contribution threshold disabled
        return true;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_minimum_light_contribution(float minimum_contribution, float contribution)
{
    if (minimum_contribution > 0.0f)
    {
        if (contribution < minimum_contribution)
            // The light doesn't contribute enough
            return false;
        else
            // The light contributes enough
            return true;
    }
    else
        // Minimum light contribution threshold disabled
        return true;
}

#endif
