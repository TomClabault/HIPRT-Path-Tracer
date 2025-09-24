/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_SAMPLING_PDF_TRIANGLES_H
#define DEVICE_LIGHT_SAMPLING_PDF_TRIANGLES_H

#include "Device/includes/BSDFSampleHitInfo.h"
#include "Device/includes/TriangleLoadUtils.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * Returns the PDF (area measure) of the light sampler for the given triangle_hit_info
 *
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit
 * 'hit_distance' is the distance to the intersection point on the hit triangle
 * 'ray_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, float light_area, ColorRGB32F light_emission)
{
    float hit_distance = 1.0f;
    float area_measure_pdf;

    // Note that for ReGIR, we cannot have the exact light PDF since ReGIR is based on RIS so we're
    // faking it with whatever base strategy ReGIR is using

    if constexpr (lightSamplingStrategy == LSS_BASE_UNIFORM)
    {
        // Surface area PDF of hitting that point on that triangle in the scene
        area_measure_pdf = 1.0f / light_area;
        area_measure_pdf /= render_data.buffers.emissive_triangles_count;
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_POWER)
    {
        area_measure_pdf = 1.0f / light_area;
        area_measure_pdf *= (light_emission.luminance() * light_area) / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
        // Faking the ReGIR PDF with the PDF of its base sampling strategy
        area_measure_pdf = pdf_of_emissive_triangle_hit_area_measure<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, light_area, light_emission);


    return area_measure_pdf;
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, int hit_primitive_index, ColorRGB32F light_emission)
{
    return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, triangle_load_area(render_data, hit_primitive_index), light_emission);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, const BSDFLightSampleRayHitInfo& light_hit_info)
{
    return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, light_hit_info.hit_prim_index, light_hit_info.hit_emission);
}

/**
 * Returns the PDF (solid angle measure) of the light sampler for the given 'light_hit_info'
 *
 * Note that for light samplers that cannot be point-evaluated (ReGIR for example: we cannot compute a RIS PDF),
 * the returned PDF is an approximation
 *
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit
 * 'hit_distance' is the distance to the intersection point on the hit triangle
 * 'to_light_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
    float light_area,
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    // abs() here to allow backfacing lights
    // Without abs() here:
    //  - We could be hitting the back of an emissive triangle (think of quad light hanging in the air)
    //  --> triangle normal not facing the same way 
    //  --> cos_angle negative
    float cosine_light_source = compute_cosine_term_at_light_source(light_surface_normal, -to_light_direction);

    float pdf_area_measure = pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, light_area, light_emission);

    return area_to_solid_angle_pdf(pdf_area_measure, hit_distance, cosine_light_source);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, int hit_primitive_index,
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data, triangle_load_area(render_data, hit_primitive_index),
        light_emission, light_surface_normal, hit_distance, to_light_direction);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, const BSDFLightSampleRayHitInfo& light_hit_info, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data,
        light_hit_info.hit_prim_index, light_hit_info.hit_emission, light_hit_info.hit_geometric_normal,
        light_hit_info.hit_distance, to_light_direction);
}

#endif
