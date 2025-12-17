/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_SAMPLING_PDF_TRIANGLES_H
#define DEVICE_LIGHT_SAMPLING_PDF_TRIANGLES_H

#include "Device/includes/BSDFSampleHitInfo.h"
#include "Device/includes/TriangleLoadUtils.h"
#include "Device/includes/LightSampling/LightTree/LightTreeATSSampling.h"

#include "HostDeviceCommon/RenderData.h"

template <int trianglePointSamplingStrategy = TrianglePointSamplingStrategy>
HIPRT_DEVICE float pdf_of_point_on_triangle_area_measure(const HIPRTRenderData& render_data, 
    float3 shading_point, float3 view_direction, float3 shading_normal,
    const DeviceUnpackedEffectiveMaterial& material,
    float3 point_on_triangle, float3 triangle_normal,
    int emissive_triangle_global_index, float light_area)
{
    if constexpr (trianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA)
    {
        return 1.0f / light_area;
    }
    else if constexpr (trianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE)
    {
        float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 0]];
        float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 1]];
        float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 2]];
        float3 to_light_direction = point_on_triangle - shading_point;

        solid_angle_triangle_t solid_angle_triangle = prepare_solid_angle_triangle_sampling(render_data, 
            vertex_A, vertex_B, vertex_C, 
            shading_point, view_direction, shading_normal, 
            material);
        float pdf_solid_angle = 1.0f / solid_angle_triangle.solid_angle;
        float to_light_distance = hippt::length(to_light_direction);
        float cosine_at_light_source = compute_cosine_term_at_light_source(triangle_normal, -to_light_direction / to_light_distance);

        return solid_angle_to_area_pdf(pdf_solid_angle, to_light_distance, cosine_at_light_source);
    }
    else if constexpr (trianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE)
    {
        float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 0]];
        float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 1]];
        float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 2]];
        ColorRGB32F triangle_emission = render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[emissive_triangle_global_index]);
        
        float3 to_light_direction = point_on_triangle - shading_point;
        float to_light_distance = hippt::length(to_light_direction);

        float solid_angle = triangle_solid_angle(vertex_A, vertex_B, vertex_C, shading_point);

        bool do_projected_solid_angle_sampling = solid_angle > render_data.render_settings.projected_solid_angle_sampling_threshold;
        if (do_projected_solid_angle_sampling)
        {
            // If the triangle is large enough in solid angle, it may be worth it to compute the heavy projected solid angle
            // stuff
            float pdf_solid_angle = projected_solid_angle_triangle_solid_angle_pdf(render_data,
                vertex_A, vertex_B, vertex_C, 
				shading_point, view_direction, shading_normal, point_on_triangle,
				ltc_lobe_probas(render_data, vertex_A, vertex_A, vertex_C, shading_point, view_direction, shading_normal, triangle_emission, material),
				material);

            return solid_angle_to_area_pdf(pdf_solid_angle, to_light_distance, compute_cosine_term_at_light_source(triangle_normal, -hippt::normalize(point_on_triangle - shading_point)));
        }
        else
        {
            // Otherwise it's not worth it and we can use the cheap solid angle (not projected) sampling
            solid_angle_triangle_t solid_angle_triangle = prepare_solid_angle_triangle_sampling(render_data, 
                vertex_A, vertex_B, vertex_C, 
                shading_point, view_direction, shading_normal, 
                material);
            float pdf_solid_angle = 1.0f / solid_angle_triangle.solid_angle;
            float cosine_at_light_source = compute_cosine_term_at_light_source(triangle_normal, -to_light_direction / to_light_distance);

            return solid_angle_to_area_pdf(1.0f / solid_angle_triangle.solid_angle, to_light_distance, cosine_at_light_source);
        }

        return 0.0f;
    }
}

/**
 * Returns the PDF (area measure) of the light sampler for the given triangle_hit_info
 *
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit
 * 'hit_distance' is the distance to the intersection point on the hit triangle
 * 'ray_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, 
    float3 shading_point, float3 view_direction, float3 shading_normal,
    const DeviceUnpackedEffectiveMaterial& material,
    float3 point_on_triangle, float3 triangle_normal,
    int emissive_triangle_global_index, float light_area, ColorRGB32F light_emission)
{
    float hit_distance = 1.0f;
    float area_measure_pdf;

    // Note that for ReGIR, we cannot have the exact light PDF since ReGIR is based on RIS so we're
    // faking it with whatever base strategy ReGIR is using

    if constexpr (lightSamplingStrategy == LSS_BASE_UNIFORM)
    {
        // Surface area PDF of hitting that point on that triangle in the scene
        area_measure_pdf = pdf_of_point_on_triangle_area_measure(render_data, 
            shading_point, view_direction, shading_normal,
            material,
            point_on_triangle, triangle_normal,
            emissive_triangle_global_index, light_area);
        area_measure_pdf /= render_data.buffers.emissive_triangles_count;
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_POWER)
    {
        area_measure_pdf = pdf_of_point_on_triangle_area_measure(render_data, 
            shading_point, view_direction, shading_normal,
            material,
            point_on_triangle, triangle_normal,
            emissive_triangle_global_index, light_area);
        area_measure_pdf *= (light_emission.luminance() * light_area) / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
    {
        area_measure_pdf = pdf_of_point_on_triangle_area_measure(render_data, 
            shading_point, view_direction, shading_normal,
            material,
            point_on_triangle, triangle_normal,
            emissive_triangle_global_index, light_area);
        area_measure_pdf *= pdf_of_emissive_triangle_light_tree_ats(render_data, shading_point, shading_normal, emissive_triangle_global_index);
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG)
    {
        area_measure_pdf = pdf_of_point_on_triangle_area_measure(render_data, 
            shading_point, view_direction, shading_normal,
            material,
            point_on_triangle, triangle_normal,
            emissive_triangle_global_index, light_area);
        area_measure_pdf *= pdf_of_emissive_triangle_light_tree_sg(render_data, shading_point, view_direction, shading_normal, material, emissive_triangle_global_index);
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
        // We should never ask that question, we can't get the PDF of ReGIR
        area_measure_pdf = 1.0e15f;
    else
        // Invalid strategy
        area_measure_pdf = 1.0e15f;


    return area_measure_pdf;
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, float3 shading_point, float3 view_direction, float3 shading_normal, 
    const DeviceUnpackedEffectiveMaterial& material,
    float3 point_on_triangle, float3 triangle_normal,
    int emissive_triangle_global_index, ColorRGB32F light_emission)
{
    return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal,
        material,
        point_on_triangle, triangle_normal,
        emissive_triangle_global_index, triangle_load_area(render_data, emissive_triangle_global_index), light_emission);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, float3 shading_point, float3 view_direction, float3 shading_normal, 
    const DeviceUnpackedEffectiveMaterial& material,
    float3 point_on_triangle, const BSDFLightSampleRayHitInfo& light_hit_info)
{
    return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal, 
        material,
		point_on_triangle, light_hit_info.hit_geometric_normal,
        light_hit_info.hit_prim_index, light_hit_info.hit_emission);
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
    float3 shading_point, float3 view_direction, float3 shading_normal, 
    const DeviceUnpackedEffectiveMaterial& material,
    int emissive_triangle_global_index, float light_area, ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    // abs() here to allow backfacing lights
    // Without abs() here:
    //  - We could be hitting the back of an emissive triangle (think of quad light hanging in the air)
    //  --> triangle normal not facing the same way 
    //  --> cos_angle negative
    float cosine_light_source = compute_cosine_term_at_light_source(light_surface_normal, -to_light_direction);

    float pdf_area_measure = pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal,
        material,
        shading_point + hit_distance * to_light_direction, light_surface_normal,
        emissive_triangle_global_index, light_area, light_emission);

    return area_to_solid_angle_pdf(pdf_area_measure, hit_distance, cosine_light_source);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
    float3 shading_point, float3 view_direction, float3 shading_normal, 
    const DeviceUnpackedEffectiveMaterial& material,
    int emissive_triangle_global_index, ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data,
        shading_point, view_direction, shading_normal,
        material,
        emissive_triangle_global_index, 
        triangle_load_area(render_data, emissive_triangle_global_index),
        light_emission, light_surface_normal, 
        hit_distance, to_light_direction);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
    float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material,
    const BSDFLightSampleRayHitInfo& light_hit_info, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data,
		shading_point, view_direction, shading_normal, material,
        light_hit_info.hit_prim_index, 
        light_hit_info.hit_emission, light_hit_info.hit_geometric_normal,
        light_hit_info.hit_distance, to_light_direction);
}

#endif
