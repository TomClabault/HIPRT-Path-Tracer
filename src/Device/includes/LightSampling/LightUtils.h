/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_UTILS_H
#define DEVICE_LIGHT_UTILS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/PDFConversion.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE HIPRT_INLINE float3 get_triangle_normal_not_normalized(const HIPRTRenderData& render_data, int triangle_index)
{
    int triangle_index_start = triangle_index * 3;

    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index_start + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index_start + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index_start + 2]];

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;

    return hippt::cross(AB, AC);
}

HIPRT_DEVICE HIPRT_INLINE float triangle_area(const HIPRTRenderData& render_data, int triangle_index)
{
    return render_data.buffers.triangles_areas[triangle_index];
}

HIPRT_DEVICE ColorRGB32F get_emission_of_triangle_from_index(const HIPRTRenderData& render_data, int triangle_index)
{
    return render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[triangle_index]);
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
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, float light_area, ColorRGB32F light_emission)
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
        area_measure_pdf *= (light_emission.luminance() * light_area) / render_data.buffers.emissives_power_alias_table.sum_elements;
    }
    else if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
        // Faking the ReGIR PDF with the PDF of its base sampling strategy
        area_measure_pdf = pdf_of_emissive_triangle_hit_area_measure<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, light_area, light_emission);


    return area_measure_pdf;
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, int hit_primitive_index, ColorRGB32F light_emission)
{
    return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, triangle_area(render_data, hit_primitive_index), light_emission);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, const BSDFLightSampleRayHitInfo& light_hit_info)
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
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
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
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, int hit_primitive_index,
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data, triangle_area(render_data, hit_primitive_index),
        light_emission, light_surface_normal, hit_distance, to_light_direction);
}

template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, const BSDFLightSampleRayHitInfo& light_hit_info, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data,
        light_hit_info.hit_prim_index, light_hit_info.hit_emission, light_hit_info.hit_geometric_normal,
        light_hit_info.hit_distance, to_light_direction);
}

/**
 * Reference: [A Low-Distortion Map Between Triangle and Square, Heitz, 2019]
 * 
 * Maps a point in a square to a point in an arbitrary triangle
 */
HIPRT_DEVICE HIPRT_INLINE float2 square_to_triangle(float& x, float& y)
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
HIPRT_DEVICE HIPRT_INLINE bool sample_point_on_generic_triangle(int global_triangle_index, const float3* vertices_positions, const int* triangles_indices, Xorshift32Generator& rng,
    float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area)
{
    float3 vertex_A = vertices_positions[triangles_indices[global_triangle_index * 3 + 0]];
    float3 vertex_B = vertices_positions[triangles_indices[global_triangle_index * 3 + 1]];
    float3 vertex_C = vertices_positions[triangles_indices[global_triangle_index * 3 + 2]];

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
    // TODO the normal length check used to be for some NaNs that occured on degenerate triangles but doesn't seem to happen anymore
    /*if (length_normal <= 1.0e-6f)
        return false;*/

    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;
    out_sample_point = random_point_on_triangle;
    out_sampled_triangle_normal = normal / length_normal;
    out_triangle_area = 0.5f * length_normal;

    return true;
}

/**
 * Samples a point uniformly on the given emissive triangle.
 * The given 'emissive_triangle_index' must come from reading the 'emissive_triangle_indices' buffer of the scene.
 *
 * Returns true if the sampling was successful, false otherwise (can fail if the triangle is way too small or degenerate)
 * 
 * !!!!! This function is a remnant of some tests. It's actually less performant than
 * sample_point_on_generic_triangle() because of more cache misses for some reason !!!!!
 */
HIPRT_DEVICE HIPRT_INLINE bool sample_point_on_emissive_triangle(int emissive_triangle_index, 
    const PrecomputedEmissiveTrianglesDataSoADevice& petd, Xorshift32Generator& rng,
    float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area)
{
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

    float3 vertex_A = petd.triangles_A[emissive_triangle_index];
    float3 AB = petd.triangles_AB[emissive_triangle_index];
    float3 AC = petd.triangles_AC[emissive_triangle_index];
    float3 normal = hippt::cross(AB, AC);

    float length_normal = hippt::length(normal);
    if (length_normal <= 1.0e-6f)
        return false;

    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;
    out_sample_point = random_point_on_triangle;
    out_sampled_triangle_normal = normal / length_normal;
    out_triangle_area = 0.5f * length_normal;

    return true;
}

/**
 * The PDF is computed in area measure
 */
HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_uniform(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSampleInformation();
        
    LightSampleInformation light_sample;

    int random_emissive_triangle_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = render_data.buffers.emissive_triangles_indices[random_emissive_triangle_index];

    float sampled_triangle_area;
    float3 sampled_triangle_normal;
    float3 random_point_on_triangle;
	if (!sample_point_on_generic_triangle(triangle_index, render_data.buffers.vertices_positions,
		render_data.buffers.triangles_indices, random_number_generator, random_point_on_triangle, sampled_triangle_normal, sampled_triangle_area))
		return LightSampleInformation();

    light_sample.emissive_triangle_index = triangle_index;
    light_sample.light_source_normal = sampled_triangle_normal;
    light_sample.light_area = sampled_triangle_area;
    light_sample.emission = render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[triangle_index]);
    light_sample.point_on_light = random_point_on_triangle;

    // PDF of that point on that triangle
    light_sample.area_measure_pdf = 1.0f / sampled_triangle_area;
    // PDF of that triangle sampled uniformly amongst all emissive triangles
    light_sample.area_measure_pdf /= render_data.buffers.emissive_triangles_count;

    return light_sample;
}

HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_power(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return LightSampleInformation();

    LightSampleInformation out_sample;

    unsigned int current_lane = hippt::current_warp_lane();
    unsigned int active_mask = hippt::warp_activemask();
    unsigned int subgroup_mask = ((1u << (render_data.render_settings.regir_settings.DEBUG_CORRELATE_rEGIR_SIZE - 1)) - 1) | (1u << render_data.render_settings.regir_settings.DEBUG_CORRELATE_rEGIR_SIZE);
    unsigned int subgroup_mask_shift = (current_lane / render_data.render_settings.regir_settings.DEBUG_CORRELATE_rEGIR_SIZE) * render_data.render_settings.regir_settings.DEBUG_CORRELATE_rEGIR_SIZE;
    unsigned int active_mask_subgroup = (active_mask & (subgroup_mask << subgroup_mask_shift)) >> subgroup_mask_shift;

    unsigned int current_lane_subgroup = current_lane % render_data.render_settings.regir_settings.DEBUG_CORRELATE_rEGIR_SIZE;
    unsigned int first_active_thread_index_subgroup = hippt::ffs(active_mask_subgroup) - 1;
    
    int random_emissive_triangle_index = -1;
    if (render_data.render_settings.DEBUG_QUICK_ALIAS_TABLE)
    {
        if (current_lane_subgroup == first_active_thread_index_subgroup)
            random_emissive_triangle_index = render_data.buffers.emissives_power_alias_table.sample(random_number_generator);
        else
        {
            if (!render_data.render_settings.DEBUG_CORRELATE_LIGHTS)
                random_emissive_triangle_index = render_data.buffers.emissives_power_alias_table.sample(random_number_generator);
            else
            {
                random_number_generator();
                random_number_generator();
            }
        }
    }
    else
        random_emissive_triangle_index = render_data.buffers.emissives_power_alias_table.sample(random_number_generator);
    
    if (render_data.render_settings.DEBUG_CORRELATE_LIGHTS)
        random_emissive_triangle_index = hippt::warp_shfl(random_emissive_triangle_index, first_active_thread_index_subgroup, render_data.render_settings.regir_settings.DEBUG_CORRELATE_rEGIR_SIZE);
    
    int triangle_index = render_data.buffers.emissive_triangles_indices[random_emissive_triangle_index];

    float sampled_triangle_area;
    float3 sampled_triangle_normal;
    float3 random_point_on_triangle;
    if (!sample_point_on_generic_triangle(triangle_index, render_data.buffers.vertices_positions,
        render_data.buffers.triangles_indices, random_number_generator, random_point_on_triangle, sampled_triangle_normal, sampled_triangle_area))
        return LightSampleInformation();

    out_sample.emissive_triangle_index = triangle_index;
    out_sample.light_source_normal = sampled_triangle_normal;
    out_sample.light_area = sampled_triangle_area;
    out_sample.emission = render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[triangle_index]);
    out_sample.point_on_light = random_point_on_triangle;

    // PDF of that point on that triangle
    out_sample.area_measure_pdf = 1.0f / sampled_triangle_area;
    // PDF of sampling that triangle according to its luminance-area
    out_sample.area_measure_pdf *= (out_sample.emission.luminance() * sampled_triangle_area) / render_data.buffers.emissives_power_alias_table.sum_elements;

    return out_sample;
}

// Forward declaration for use in 'sample_one_emissive_triangle_regir' below
template <int samplingStrategy>
HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle(const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    Xorshift32Generator& random_number_generator);

/**
 * For ReGIR's pairwise MIS, the canonical sampling technique is the sampling technique that produces samples
 * without cosine terms / visibility terms / ...
 */

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, const ReGIRGridFillSurface& surface, float PDF_normalization, float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, Xorshift32Generator& random_number_generator)
{
    float sample_PDF_unnormalized;
    if constexpr (canonicalPDF)
        sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, surface, emission, light_source_normal, point_on_light, random_number_generator);
    else
        sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, surface, emission, light_source_normal, point_on_light, random_number_generator);

    return sample_PDF_unnormalized / PDF_normalization;
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, const ReGIRGridFillSurface& surface, unsigned int grid_cell_index, bool primary_hit,
    float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, Xorshift32Generator& random_number_generator)
{
    float RIS_integral;
    if constexpr (canonicalPDF)
        RIS_integral = render_data.render_settings.regir_settings.get_canonical_pre_integration_factor(grid_cell_index, primary_hit);
    else
        RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(grid_cell_index, primary_hit);
    if (RIS_integral == 0.0f)
        RIS_integral = 1.0f;
    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
        RIS_integral = 1.0f;

    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, surface, RIS_integral, point_on_light, light_source_normal, emission, random_number_generator);
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, float3 point_on_light, float3 light_source_normal, ColorRGB32F emission, unsigned int grid_cell_index, bool primary_hit, Xorshift32Generator& random_number_generator)
{
    if (emission.is_black())
        return 0.0f;

    ReGIRGridFillSurface surface = ReGIR_get_cell_surface(render_data, grid_cell_index, primary_hit);
    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, surface, grid_cell_index, primary_hit, point_on_light, light_source_normal, emission, random_number_generator);
}

template <bool canonicalPDF>
HIPRT_DEVICE float ReGIR_get_reservoir_sample_ReGIR_PDF(const HIPRTRenderData& render_data, const ReGIRReservoir& reservoir, unsigned int grid_cell_index, bool primary_hit, Xorshift32Generator& random_number_generator)
{
    if (reservoir.UCW <= 0.0f)
        return 0.0f;

    float3 point_on_light = reservoir.sample.point_on_light;
    float3 light_source_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_index));
    ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, reservoir.sample.emissive_triangle_index);

    return ReGIR_get_reservoir_sample_ReGIR_PDF<canonicalPDF>(render_data, point_on_light, light_source_normal, emission, grid_cell_index, primary_hit, random_number_generator);
}

HIPRT_DEVICE float ReGIR_get_reservoir_sample_BSDF_PDF(const HIPRTRenderData& render_data,
	float3 point_on_light, float3 light_source_normal, ColorRGB32F emission,
    float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, BSDFIncidentLightInfo incident_light_info, RayPayload& ray_payload, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
    if (emission.is_black())
        return 0.0f;

    float3 to_light_direction = point_on_light - shading_point;
    float distance_to_light = hippt::length(to_light_direction);
    to_light_direction /= distance_to_light; // Normalization

    float bsdf_pdf;
    BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, to_light_direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
    ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

    hiprtRay shadow_ray;
    shadow_ray.origin = shading_point;
    shadow_ray.direction = to_light_direction;
    // bsdf_pdf *= !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light, last_hit_primitive_index, ray_payload.bounce, random_number_generator);

    float area_measure_bsdf_pdf = solid_angle_to_area_pdf(bsdf_pdf, distance_to_light, compute_cosine_term_at_light_source(light_source_normal, -to_light_direction));

    return area_measure_bsdf_pdf;
}

HIPRT_DEVICE float ReGIR_get_reservoir_sample_BSDF_PDF(const HIPRTRenderData& render_data, const ReGIRReservoir& reservoir,
    float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, BSDFIncidentLightInfo incident_light_info, RayPayload& ray_payload, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
    if (reservoir.UCW <= 0.0f)
        return 0.0f;

    float3 point_on_light = reservoir.sample.point_on_light;
    float3 light_source_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_index));
    ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, reservoir.sample.emissive_triangle_index);

    return ReGIR_get_reservoir_sample_BSDF_PDF(render_data,
        point_on_light, light_source_normal, emission, 
        view_direction, shading_point, shading_normal, geometric_normal, incident_light_info, ray_payload, last_hit_primitive_index, random_number_generator);
}

struct ReGIRPairwiseMIS
{
    HIPRT_DEVICE float compute_MIS_weight_normalization(const HIPRTRenderData& render_data, unsigned int valid_non_canonical_neighbors)
    {
        unsigned int number_of_samples = 0;
        number_of_samples += valid_non_canonical_neighbors * render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor; // non canonical samples
        if (number_of_samples == 0)
            return 0.0f;

        return 1.0f / number_of_samples;
    }

    /*HIPRT_DEVICE float compute_MIS_weight_for_BSDF_sample(const HIPRTRenderData& render_data,
		const ReGIRGridFillSurface& center_grid_cell_surface,
        float bsdf_pdf_area_measure,
        float mis_weight_normalization,

        float non_canonical_RIS_integral_center_grid_cell, float canonical_RIS_integral_center_grid_cell,

		float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal, RayPayload& ray_payload, int last_hit_primitive_index,
        BSDFIncidentLightInfo incident_light_info,

        const LightSampleInformation& light_sample,
        Xorshift32Generator& random_number_generator)
    {
        float non_canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, center_grid_cell_surface, non_canonical_RIS_integral_center_grid_cell, light_sample.point_on_light, light_sample.light_source_normal, light_sample.emission, random_number_generator);
        float canonical_sample_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, center_grid_cell_surface, canonical_RIS_integral_center_grid_cell, light_sample.point_on_light, light_sample.light_source_normal, light_sample.emission, random_number_generator);

        float mis_weight = mis_weight_normalization * (bsdf_pdf_area_measure / (bsdf_pdf_area_measure + non_canonical_PDF * mis_weight_normalization + canonical_sample_PDF * mis_weight_normalization));

        return mis_weight;
    }*/

    /*HIPRT_DEVICE void sum_BSDF_sample_to_canonical_weights(const HIPRTRenderData& render_data, 
        const ReGIRReservoir& canonical_technique_1_reservoir, const ReGIRReservoir& canonical_technique_2_reservoir,
        float3 canonical_technique_3_point_on_light, float3 canonical_technique_3_light_normal, ColorRGB32F canonical_technique_3_emission,

        float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_1_canonical_reservoir_3_pdf,
        float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_3_pdf,
        float canonical_technique_3_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_3_pdf,
        float mis_weight_normalization,

        float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, BSDFIncidentLightInfo incident_light_info, RayPayload& ray_payload, int last_hit_primitive_index,
        Xorshift32Generator& random_number_generator)
    {
        float BSDF_technique_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, canonical_technique_1_reservoir, view_direction, shading_point, shading_normal, geometric_normal, incident_light_info, ray_payload, last_hit_primitive_index, random_number_generator);
        m_sum_canonical_weight_1 += canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization / (BSDF_technique_canonical_reservoir_1_pdf + canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_1_canonical_reservoir_3_pdf * mis_weight_normalization);

        float BSDF_technique_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, canonical_technique_2_reservoir, view_direction, shading_point, shading_normal, geometric_normal, incident_light_info, ray_payload, last_hit_primitive_index, random_number_generator);
        m_sum_canonical_weight_2 += canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization / (BSDF_technique_canonical_reservoir_2_pdf + canonical_technique_1_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_3_pdf * mis_weight_normalization);

        float BSDF_technique_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, canonical_technique_3_point_on_light, canonical_technique_3_light_normal, canonical_technique_3_emission, view_direction, shading_point, shading_normal, geometric_normal, incident_light_info, ray_payload, last_hit_primitive_index, random_number_generator);
        m_sum_canonical_weight_3 += canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization / (BSDF_technique_canonical_reservoir_3_pdf + canonical_technique_1_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization);
    }*/

    HIPRT_DEVICE void sum_non_canonical_sample_to_canonical_weights(const HIPRTRenderData& render_data, 
        const ReGIRReservoir& canonical_technique_1_reservoir, const ReGIRReservoir& canonical_technique_2_reservoir,
        float3 canonical_technique_3_point_on_light, float3 canonical_technique_3_light_normal, ColorRGB32F canonical_technique_3_emission,

        float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_1_canonical_reservoir_3_pdf,
        float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_3_pdf,
        float canonical_technique_3_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_3_pdf,
        float mis_weight_normalization,
        
        float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, RayPayload& ray_payload, int last_hit_primitive_index,

        unsigned int neighbor_grid_cell_index, bool is_primary_hit,
        Xorshift32Generator& random_number_generator)
    {
        float non_canonical_neighbor_technique_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, canonical_technique_1_reservoir, neighbor_grid_cell_index, is_primary_hit, random_number_generator);
        m_sum_canonical_weight_1 += canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_1_pdf + canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_1_pdf * mis_weight_normalization);

        float non_canonical_neighbor_technique_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, canonical_technique_2_reservoir, neighbor_grid_cell_index, is_primary_hit, random_number_generator);
        m_sum_canonical_weight_2 += canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_2_pdf + canonical_technique_1_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_2_pdf * mis_weight_normalization);

        float non_canonical_neighbor_technique_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, canonical_technique_3_point_on_light, canonical_technique_3_light_normal, canonical_technique_3_emission, neighbor_grid_cell_index, is_primary_hit, random_number_generator);
        //float non_canonical_neighbor_technique_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, canonical_technique_3_point_on_light, canonical_technique_3_light_normal, canonical_technique_3_emission, view_direction, shading_point, shading_normal, geometric_normal, BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index, random_number_generator);
        m_sum_canonical_weight_3 += canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_3_pdf + canonical_technique_1_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization);
    }

  //  HIPRT_DEVICE float compute_MIS_weight_for_canonical_sample(const HIPRTRenderData& render_data,
  //      const ReGIRReservoir& canonical_technique_1_reservoir, const ReGIRReservoir& canonical_technique_2_reservoir,
		//const ReGIRGridFillSurface& center_grid_cell_surface,
  //      float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_1_canonical_reservoir_2_pdf,
  //      float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_2_pdf,
  //      float mis_weight_normalization,

  //      float non_canonical_RIS_integral_center_grid_cell, float canonical_RIS_integral_center_grid_cell,

  //      float canonical_sample_PDF, bool is_primary_hit,

  //      float3 shading_point, ColorRGB32F light_sample_emission, float3 light_source_normal, float3 point_on_light, unsigned int neighbor_grid_cell_index,
  //      Xorshift32Generator& random_number_generator,
  //      unsigned int valid_non_canonical_neighbors)
  //  {
  //      float non_canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, center_grid_cell_surface, non_canonical_RIS_integral_center_grid_cell, point_on_light, light_source_normal, light_sample_emission, random_number_generator);
  //      float canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, center_grid_cell_surface, canonical_RIS_integral_center_grid_cell, point_on_light, light_source_normal, light_sample_emission, random_number_generator);

  //      float mis_weight = mis_weight_normalization * (canonical_sample_PDF / (canonical_sample_PDF + non_canonical_PDF * mis_weight_normalization + canonical_PDF * mis_weight_normalization));

  //      // Summing the weights for the canonical MIS weight computation
  //      float canonical_neighbor_technique_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, canonical_technique_1_reservoir, neighbor_grid_cell_index, is_primary_hit, random_number_generator);
  //      m_sum_canonical_weight_1 += canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization / (canonical_neighbor_technique_canonical_reservoir_1_pdf + canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_1_pdf * mis_weight_normalization);

  //      float canonical_neighbor_technique_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, canonical_technique_2_reservoir, neighbor_grid_cell_index, is_primary_hit, random_number_generator);
  //      m_sum_canonical_weight_2 += canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization / (canonical_neighbor_technique_canonical_reservoir_2_pdf + canonical_technique_1_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization);

  //      return mis_weight;
  //  }

    HIPRT_DEVICE float compute_MIS_weight_for_non_canonical_sample(const HIPRTRenderData& render_data,
        float3 sample_point_on_light, float3 sample_light_source_normal, ColorRGB32F sample_emission,

        const ReGIRReservoir& canonical_technique_1_reservoir, const ReGIRReservoir& canonical_technique_2_reservoir,
        float3 canonical_technique_3_point_on_light, float3 canonical_technique_3_light_normal, ColorRGB32F canonical_technique_3_emission,
        const ReGIRGridFillSurface& center_grid_cell_surface,

        float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_1_canonical_reservoir_3_pdf,
        float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_3_pdf,
        float canonical_technique_3_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_3_pdf,
        float mis_weight_normalization,

        float non_canonical_RIS_integral_center_grid_cell, float canonical_RIS_integral_center_grid_cell,
        float non_canonical_sample_PDF,

        unsigned int neighbor_grid_cell_index,

        float3 view_direction, float3 shading_point, float3 shading_normal, float3 geometric_normal, RayPayload& ray_payload, int last_hit_primitive_index,

        Xorshift32Generator& random_number_generator)
    {
        // TODO these functions here recompute the grid fill target function but we already have it outside of this function call for streaming the reservoir weight
        float non_canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, center_grid_cell_surface, non_canonical_RIS_integral_center_grid_cell, sample_point_on_light, sample_light_source_normal, sample_emission, random_number_generator);
        float canonical_PDF = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, center_grid_cell_surface, canonical_RIS_integral_center_grid_cell, sample_point_on_light, sample_light_source_normal, sample_emission, random_number_generator);
        // TODO do we have that PDF thanks to the evaluation of the target function at the shading point?

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
        float bsdf_PDF = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, sample_point_on_light, sample_light_source_normal, sample_emission, view_direction, shading_point, shading_normal, geometric_normal, BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index, random_number_generator);
#else
        float bsdf_PDF = 0.0f;
#endif

        float mis_weight = mis_weight_normalization * (non_canonical_sample_PDF / (non_canonical_sample_PDF + non_canonical_PDF * mis_weight_normalization + canonical_PDF * mis_weight_normalization + bsdf_PDF * mis_weight_normalization));

        // Summing the weights for the canonical MIS weight computation
        // TODO all of these functions here reload the neighbor surface but we already have outside of this function call
        float non_canonical_neighbor_technique_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, canonical_technique_1_reservoir, neighbor_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
        m_sum_canonical_weight_1 += canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_1_pdf + canonical_technique_1_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_1_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_1_pdf * mis_weight_normalization);

        // TODO all of these functions here reload the neighbor surface but we already have outside of this function call
        float non_canonical_neighbor_technique_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, canonical_technique_2_reservoir, neighbor_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
        m_sum_canonical_weight_2 += canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_2_pdf + canonical_technique_1_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_2_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_2_pdf * mis_weight_normalization);

        // TODO all of these functions here reload the neighbor surface but we already have outside of this function call
        float non_canonical_neighbor_technique_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, canonical_technique_3_point_on_light, canonical_technique_3_light_normal, canonical_technique_3_emission, neighbor_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
        m_sum_canonical_weight_3 += canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization / (non_canonical_neighbor_technique_canonical_reservoir_3_pdf + canonical_technique_1_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_2_canonical_reservoir_3_pdf * mis_weight_normalization + canonical_technique_3_canonical_reservoir_3_pdf * mis_weight_normalization);

        return mis_weight;
    }

    HIPRT_DEVICE float get_canonical_MIS_weight_1(float canonical_technique_1_canonical_reservoir_1_pdf, float canonical_technique_2_canonical_reservoir_1_pdf, float canonical_technique_3_canonical_reservoir_1_pdf, float mis_weight_normalization)
    {
        if (mis_weight_normalization == 0.0f)
            // We only have the canonical techniques available, we're going to go for a balance heuristic between them
            return canonical_technique_1_canonical_reservoir_1_pdf / (canonical_technique_1_canonical_reservoir_1_pdf + canonical_technique_2_canonical_reservoir_1_pdf + canonical_technique_3_canonical_reservoir_1_pdf);

        return m_sum_canonical_weight_1 * mis_weight_normalization;
    }

    HIPRT_DEVICE float get_canonical_MIS_weight_2(float canonical_technique_1_canonical_reservoir_2_pdf, float canonical_technique_2_canonical_reservoir_2_pdf, float canonical_technique_3_canonical_reservoir_2_pdf, float mis_weight_normalization)
    {
        if (mis_weight_normalization == 0.0f)
            // We only have the canonical techniques available, we're going to go for a balance heuristic between them
            return canonical_technique_2_canonical_reservoir_2_pdf / (canonical_technique_1_canonical_reservoir_2_pdf + canonical_technique_2_canonical_reservoir_2_pdf + canonical_technique_3_canonical_reservoir_2_pdf);

        return m_sum_canonical_weight_2 * mis_weight_normalization;
    }

    HIPRT_DEVICE float get_canonical_MIS_weight_3(float canonical_technique_1_canonical_reservoir_3_pdf, float canonical_technique_2_canonical_reservoir_3_pdf, float canonical_technique_3_canonical_reservoir_3_pdf, float mis_weight_normalization)
    {
        if (mis_weight_normalization == 0.0f)
            // We only have the canonical techniques available, we're going to go for a balance heuristic between them
            return canonical_technique_3_canonical_reservoir_3_pdf / (canonical_technique_1_canonical_reservoir_3_pdf + canonical_technique_2_canonical_reservoir_3_pdf + canonical_technique_3_canonical_reservoir_3_pdf);

        return m_sum_canonical_weight_3 * mis_weight_normalization;
    }

    // 1st is non-canonical samples
    float m_sum_canonical_weight_1 = 0.0f;
    // 2nd technique is canonical samples
    float m_sum_canonical_weight_2 = 0.0f;
    // 3rd technique is BSDF samples
    float m_sum_canonical_weight_3 = 0.0f;
};

HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_regir(
    const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    bool& out_need_fallback_sampling,
    Xorshift32Generator& random_number_generator)
{
    // Starting with this at true and if we find a single good neighbor,
    // this will be set to false
    out_need_fallback_sampling = true;

    float3 selected_point_on_light;
    float3 selected_light_source_normal;
    float selected_light_source_area = 0.0f;
    BSDFIncidentLightInfo selected_incident_light_info = BSDFIncidentLightInfo::NO_INFO;
    ColorRGB32F selected_emission;

    ReGIRReservoir out_reservoir;

    // Some random seed to generate to positions of the neighbors (when jittering)
    // XORing here because not XORing was causing RNG correlations issues...
    // not sure how that works but more randomness here seems to be getting rid of those correlations issues
    unsigned neighbor_rng_seed = random_number_generator.xorshift32() ^ random_number_generator.xorshift32();
    unsigned non_cano_neighbor_rng_seed = neighbor_rng_seed ^ random_number_generator.xorshift32();
    Xorshift32Generator non_canonical_neighbor_rng(non_cano_neighbor_rng_seed);
    Xorshift32Generator neighbor_rng(neighbor_rng_seed);

    unsigned int valid_non_canonical_neighbors = 0;
    for (int neighbor = 0; neighbor < render_data.render_settings.regir_settings.shading.number_of_neighbors; neighbor++)
    {
        unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<false>(
            shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
            render_data.render_settings.regir_settings.shading.get_do_cell_jittering(ray_payload.bounce == 0),
            render_data.render_settings.regir_settings.shading.jittering_radius, non_canonical_neighbor_rng);
        if (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
            // Not a valid neighbor
            continue;
        else
            valid_non_canonical_neighbors++;
    }
    // Resetting the seed after the counting of the neighbors
    non_canonical_neighbor_rng.m_state.seed = non_cano_neighbor_rng_seed;

#if ReGIR_ShadingResamplingDoMISPairwiseMIS
    {
        ReGIRPairwiseMIS pairwise;

        unsigned int canonical_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<true>(
            shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
            false,
            render_data.render_settings.regir_settings.shading.jittering_radius, neighbor_rng);

        ReGIRReservoir canonical_technique_1_reservoir;
        ReGIRReservoir canonical_technique_2_reservoir;
        LightSampleInformation canonical_technique_3_sample;
        BSDFIncidentLightInfo canonical_technique_3_sample_ili = BSDFIncidentLightInfo::NO_INFO;
        ReGIRGridFillSurface center_cell_surface;

        //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2
        //ReGIRReservoir deb_g_reseriovir;
        //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2

        float canonical_technique_1_canonical_reservoir_1_pdf = 0.0f;
        float canonical_technique_1_canonical_reservoir_2_pdf = 0.0f;
        float canonical_technique_1_canonical_reservoir_3_pdf = 0.0f;
        float canonical_technique_2_canonical_reservoir_1_pdf = 0.0f;
        float canonical_technique_2_canonical_reservoir_2_pdf = 0.0f;
        float canonical_technique_2_canonical_reservoir_3_pdf = 0.0f;
        float canonical_technique_3_canonical_reservoir_1_pdf = 0.0f;
        float canonical_technique_3_canonical_reservoir_2_pdf = 0.0f;
        float canonical_technique_3_canonical_reservoir_3_pdf = 0.0f;
        float mis_weight_normalization = pairwise.compute_MIS_weight_normalization(render_data, valid_non_canonical_neighbors);

        float non_canonical_RIS_integral_center_grid_cell;
        float canonical_RIS_integral_center_grid_cell;

        // Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
        // that cannot be resolved
        if (canonical_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
        {
            // We found at least one good sample so we're not going to need a fallback on another light sampling strategy than ReGIR
            out_need_fallback_sampling = false;

            // Producing the canonical techniques samples
            {
                canonical_technique_1_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
                canonical_technique_2_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<true>(canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);

                if constexpr (ReGIR_ShadingResamplingDoBSDFMIS)
                {
                    float bsdf_sample_pdf;
                    float3 sampled_bsdf_direction;

                    BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, make_float3(0.0f, 0.0f, 0.0f), canonical_technique_3_sample_ili, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
                    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

                    bool intersection_found = false;
                    BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
                    if (bsdf_sample_pdf > 0.0f)
                    {
                        hiprtRay new_ray;
                        new_ray.origin = shading_point;
                        new_ray.direction = sampled_bsdf_direction;

                        intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index, ray_payload.bounce, random_number_generator);

                        // Checking that we did hit something and if we hit something,
                        // it needs to be emissive
                        if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black())
                        {
                            // if (hippt::is_pixel_index(713, render_data.render_settings.render_resolution.y - 1 - 391))
                            // {
                            //     printf("kewl: %d\n", canonical_technique_3_sample_ili);
                            //     printf("specular | roughness: %f | %f\n", ray_payload.material.specular, ray_payload.material.roughness);
                            //     printf("\n");
                            // }

                            canonical_technique_3_sample.emission = shadow_light_ray_hit_info.hit_emission;
                            canonical_technique_3_sample.emissive_triangle_index = shadow_light_ray_hit_info.hit_prim_index;
                            canonical_technique_3_sample.light_area = triangle_area(render_data, shadow_light_ray_hit_info.hit_prim_index);
                            canonical_technique_3_sample.light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                            canonical_technique_3_sample.point_on_light = shading_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;

                            // We want ReGIR to produce PDFs that are in area measure so we're converting from solid angle to area measure here
                            canonical_technique_3_canonical_reservoir_3_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                        }
                    }
                }

                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2
                //// deb_g_reseriovir = canonical_technique_2_reservoir;

                //deb_g_reseriovir.sample.emissive_triangle_index = canonical_technique_3_sample.emissive_triangle_index;
                //deb_g_reseriovir.sample.point_on_light = canonical_technique_3_sample.point_on_light;
                //deb_g_reseriovir.UCW = canonical_technique_3_sample.area_measure_pdf == 0.0f ? 0.0f : 1.0f / canonical_technique_3_sample.area_measure_pdf;
                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2
                //// 
                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2
                //canonical_technique_1_reservoir = deb_g_reseriovir;
                //canonical_technique_2_reservoir = deb_g_reseriovir;

                //if (deb_g_reseriovir.UCW <= 0.0f)
                //    return LightSampleInformation();

                //canonical_technique_3_sample.emission = get_emission_of_triangle_from_index(render_data, deb_g_reseriovir.sample.emissive_triangle_index);
                //canonical_technique_3_sample.emissive_triangle_index = deb_g_reseriovir.sample.emissive_triangle_index;
                //canonical_technique_3_sample.light_area = triangle_area(render_data, deb_g_reseriovir.sample.emissive_triangle_index);
                //canonical_technique_3_sample.light_source_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, deb_g_reseriovir.sample.emissive_triangle_index));
                //canonical_technique_3_sample.point_on_light = deb_g_reseriovir.sample.point_on_light;

                //canonical_technique_3_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, canonical_technique_3_sample.point_on_light, canonical_technique_3_sample.light_source_normal, canonical_technique_3_sample.emission, view_direction, shading_point, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index, random_number_generator);
                //// TODO DEBUG REMOVE THIS //////////////////
            }

            // Computing all the PDFs of the canonical techniques that we're going to need for pairwise MIS
            {
                if (canonical_technique_1_reservoir.UCW > 0.0f)
                {
                    float3 point_on_light_1 = canonical_technique_1_reservoir.sample.point_on_light;
                    float3 light_source_normal_1 = hippt::normalize(get_triangle_normal_not_normalized(render_data, canonical_technique_1_reservoir.sample.emissive_triangle_index));
                    ColorRGB32F emission_1 = get_emission_of_triangle_from_index(render_data, canonical_technique_1_reservoir.sample.emissive_triangle_index);

                    // TODO we already the canonical / non-canonical PDF normalization (fetched below) so we can use them because otherwise, that function fetches them again
                    canonical_technique_1_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, point_on_light_1, light_source_normal_1, emission_1, canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
                    canonical_technique_2_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, point_on_light_1, light_source_normal_1, emission_1, canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
                    canonical_technique_3_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, point_on_light_1, light_source_normal_1, emission_1, view_direction, shading_point, shading_normal, geometric_normal, BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index, random_number_generator);
#endif
                }

                if (canonical_technique_2_reservoir.UCW > 0.0f)
                {
                    float3 point_on_light_2 = canonical_technique_2_reservoir.sample.point_on_light;
                    float3 light_source_normal_2 = hippt::normalize(get_triangle_normal_not_normalized(render_data, canonical_technique_2_reservoir.sample.emissive_triangle_index));
                    ColorRGB32F emission_2 = get_emission_of_triangle_from_index(render_data, canonical_technique_2_reservoir.sample.emissive_triangle_index);

                    canonical_technique_1_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, point_on_light_2, light_source_normal_2, emission_2, canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
                    canonical_technique_2_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, point_on_light_2, light_source_normal_2, emission_2, canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
                    canonical_technique_3_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(render_data, point_on_light_2, light_source_normal_2, emission_2, view_direction, shading_point, shading_normal, geometric_normal, BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index, random_number_generator);
#endif
                }

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
                if (!canonical_technique_3_sample.emission.is_black())
                {
                    float3 point_on_light_3 = canonical_technique_3_sample.point_on_light;
                    float3 light_source_normal_3 = canonical_technique_3_sample.light_source_normal;
                    ColorRGB32F emission_3 = canonical_technique_3_sample.emission;

                    canonical_technique_1_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, point_on_light_3, light_source_normal_3, emission_3, canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
                    canonical_technique_2_canonical_reservoir_3_pdf = ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, point_on_light_3, light_source_normal_3, emission_3, canonical_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
                    // This one has already been computed when sampling the BSDF sample
                    // canonical_technique_3_canonical_reservoir_3_pdf....
                }
#endif
            }

            {
                non_canonical_RIS_integral_center_grid_cell = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(canonical_grid_cell_index, ray_payload.bounce == 0);
                if (non_canonical_RIS_integral_center_grid_cell == 0.0f)
                    non_canonical_RIS_integral_center_grid_cell = 1.0f;
                if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                    non_canonical_RIS_integral_center_grid_cell = 1.0f;

                canonical_RIS_integral_center_grid_cell = render_data.render_settings.regir_settings.get_canonical_pre_integration_factor(canonical_grid_cell_index, ray_payload.bounce == 0);
                if (canonical_RIS_integral_center_grid_cell == 0.0f)
                    canonical_RIS_integral_center_grid_cell = 1.0f;
                if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                    canonical_RIS_integral_center_grid_cell = 1.0f;
            }

			center_cell_surface = ReGIR_get_cell_surface(render_data, canonical_grid_cell_index, ray_payload.bounce == 0);
        }
        else
        {
            // The center grid cell is invalid (must be because of hash grid collisions that couldn't be resolved)
            out_need_fallback_sampling = true;

            return LightSampleInformation();
        }

        for (int neighbor = 0; neighbor < render_data.render_settings.regir_settings.shading.number_of_neighbors; neighbor++)
        {
            unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<false>(
                shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
                render_data.render_settings.regir_settings.shading.get_do_cell_jittering(ray_payload.bounce == 0),
                render_data.render_settings.regir_settings.shading.jittering_radius, non_canonical_neighbor_rng);
            if (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                // Couldn't find a valid neighbor
                continue;
            else
                out_need_fallback_sampling = false;

            float neighbor_RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(neighbor_grid_cell_index, ray_payload.bounce == 0);
            if (neighbor_RIS_integral == 0.0f)
                neighbor_RIS_integral = 1.0f;
            if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                neighbor_RIS_integral = 1.0f;

            for (int i = 0; i < render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor; i++)
            {
                // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
                ReGIRReservoir non_canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(neighbor_grid_cell_index, ray_payload.bounce == 0, neighbor_rng);
                
                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2
                //non_canonical_reservoir = deb_g_reseriovir;
                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2

                if (non_canonical_reservoir.UCW <= 0.0f)
                {
                    // No valid sample in that reservoir

                    pairwise.sum_non_canonical_sample_to_canonical_weights(render_data, 
                        canonical_technique_1_reservoir, canonical_technique_2_reservoir,
                        canonical_technique_3_sample.point_on_light, canonical_technique_3_sample.light_source_normal, canonical_technique_3_sample.emission,

                        canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_1_canonical_reservoir_2_pdf, canonical_technique_1_canonical_reservoir_3_pdf,
                        canonical_technique_2_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_3_pdf,
                        canonical_technique_3_canonical_reservoir_1_pdf, canonical_technique_3_canonical_reservoir_2_pdf, canonical_technique_3_canonical_reservoir_3_pdf,
                        mis_weight_normalization,

                        view_direction, shading_point, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index,
                        neighbor_grid_cell_index, ray_payload.bounce == 0, random_number_generator);
                    continue;
                }

                float3 point_on_light = non_canonical_reservoir.sample.point_on_light;
                float3 light_source_normal = get_triangle_normal_not_normalized(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
                float light_source_area = hippt::length(light_source_normal) * 0.5f;
                light_source_normal /= light_source_area * 2.0f;
                ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, non_canonical_reservoir.sample.emissive_triangle_index);

                float target_function = ReGIR_shading_evaluate_target_function<
                    ReGIR_ShadingResamplingTargetFunctionVisibility,
                    ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal,
                        last_hit_primitive_index, ray_payload,
                        point_on_light, light_source_normal,
                        emission, random_number_generator);

                float non_canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_grid_cell_index, ray_payload.bounce == 0, 
                    emission, light_source_normal, point_on_light, random_number_generator);
                float current_sample_PDF = non_canonical_sample_PDF_unnormalized / neighbor_RIS_integral;
                float mis_weight = pairwise.compute_MIS_weight_for_non_canonical_sample(render_data,
                    point_on_light, light_source_normal, emission,

                    canonical_technique_1_reservoir, canonical_technique_2_reservoir,
                    canonical_technique_3_sample.point_on_light, canonical_technique_3_sample.light_source_normal, canonical_technique_3_sample.emission,
                    center_cell_surface,

                    canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_1_canonical_reservoir_2_pdf, canonical_technique_1_canonical_reservoir_3_pdf,
                    canonical_technique_2_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_3_pdf,
                    canonical_technique_3_canonical_reservoir_1_pdf, canonical_technique_3_canonical_reservoir_2_pdf, canonical_technique_3_canonical_reservoir_3_pdf,

                    mis_weight_normalization,

                    non_canonical_RIS_integral_center_grid_cell, canonical_RIS_integral_center_grid_cell, current_sample_PDF,
                    neighbor_grid_cell_index,
                    view_direction, shading_point, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index,
                    random_number_generator);

                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2
                //float mis_weight_1 = pairwise.get_canonical_MIS_weight_1(canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_1_pdf, canonical_technique_3_canonical_reservoir_1_pdf, mis_weight_normalization);
                //float mis_weight_2 = pairwise.get_canonical_MIS_weight_2(canonical_technique_1_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_2_pdf, canonical_technique_3_canonical_reservoir_2_pdf, mis_weight_normalization);
                //float mis_weight_3 = pairwise.get_canonical_MIS_weight_3(canonical_technique_1_canonical_reservoir_3_pdf, canonical_technique_2_canonical_reservoir_3_pdf, canonical_technique_3_canonical_reservoir_3_pdf, mis_weight_normalization);
                //// TODO DEBUG REMOVE THIS ////////////////// Everyone uses the _2

                //if (hippt::abs((mis_weight + mis_weight_1 + mis_weight_2 + mis_weight_3 - 1.0f) > 1.0e-3f))
                //{
                //    std::cout << "Sum: " << mis_weight + mis_weight_1 + mis_weight_2 + mis_weight_3 << std::endl;
                //    std::cout << std::endl;
                //}

                if (out_reservoir.stream_reservoir(mis_weight, target_function, non_canonical_reservoir, random_number_generator))
                {
                    selected_point_on_light = point_on_light;
                    selected_light_source_normal = light_source_normal;
                    selected_light_source_area = light_source_area;
                    selected_emission = emission;
                }
            }
        }

        if (canonical_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
        {
            if (canonical_technique_1_reservoir.UCW > 0.0f && canonical_technique_1_reservoir.UCW != ReGIRReservoir::UNDEFINED_UCW)
            {
                ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, canonical_technique_1_reservoir.sample.emissive_triangle_index);

                float3 point_on_light = canonical_technique_1_reservoir.sample.point_on_light;
                float3 light_source_normal = get_triangle_normal_not_normalized(render_data, canonical_technique_1_reservoir.sample.emissive_triangle_index);
                float light_source_area = hippt::length(light_source_normal) * 0.5f;
                light_source_normal /= light_source_area * 2.0f;

                {
                    // Adding visibility in the canonical sample target function's if we have visibility reuse
                    // (or visibility in the grid fill target function) because otherwise this canonical sample
                    // will kill all the benefits of the visibility reuse
                    //
                    // TLDR is that this is pretty much necessary for good visibility reuse quality
                    float target_function = ReGIR_shading_evaluate_target_function<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal,
                        last_hit_primitive_index, ray_payload,
                        point_on_light, light_source_normal,
                        emission, random_number_generator);

                    float RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(canonical_grid_cell_index, ray_payload.bounce == 0);
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;
                    float non_canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, canonical_grid_cell_index, ray_payload.bounce == 0,
                        emission, light_source_normal, point_on_light, random_number_generator);
                    float non_canonical_sample_PDF = non_canonical_sample_PDF_unnormalized / RIS_integral;

                    float mis_weight = pairwise.get_canonical_MIS_weight_1(canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_1_pdf, canonical_technique_3_canonical_reservoir_1_pdf, mis_weight_normalization);

                    if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_technique_1_reservoir, random_number_generator))
                    {
                        selected_point_on_light = point_on_light;
                        selected_light_source_normal = light_source_normal;
                        selected_light_source_area = light_source_area;
                        selected_emission = emission;
                    }
                }
            }
        }

        // Incorporating a canonical candidate if doing visibility reuse because visibility reuse
        // may cause the grid cell to produce no valid reservoir at all so we need canonical samples to
        // cover those cases for unbiased results
        // 
        // Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
        // that cannot be resolved
        if (canonical_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
        {
            ReGIRReservoir canonical_reservoir = canonical_technique_2_reservoir;

            if (canonical_reservoir.UCW > 0.0f && canonical_reservoir.UCW != ReGIRReservoir::UNDEFINED_UCW)
            {
                ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, canonical_reservoir.sample.emissive_triangle_index);

                float3 point_on_light = canonical_reservoir.sample.point_on_light;
                float3 light_source_normal = get_triangle_normal_not_normalized(render_data, canonical_reservoir.sample.emissive_triangle_index);
                float light_source_area = hippt::length(light_source_normal) * 0.5f;
                light_source_normal /= light_source_area * 2.0f;

                {
                    // Adding visibility in the canonical sample target function's if we have visibility reuse
                    // (or visibility in the grid fill target function) because otherwise this canonical sample
                    // will kill all the benefits of the visibility reuse
                    //
                    // TLDR is that this is pretty much necessary for good visibility reuse quality
                    float target_function = ReGIR_shading_evaluate_target_function<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal,
                        last_hit_primitive_index, ray_payload,
                        point_on_light, light_source_normal,
                        emission, random_number_generator);

                    float RIS_integral = render_data.render_settings.regir_settings.get_canonical_pre_integration_factor(canonical_grid_cell_index, ray_payload.bounce == 0);
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;
                    float canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, canonical_grid_cell_index, ray_payload.bounce == 0,
                        emission, light_source_normal, point_on_light, random_number_generator);
                    float canonical_sample_PDF = canonical_sample_PDF_unnormalized / RIS_integral;

                    float mis_weight = pairwise.get_canonical_MIS_weight_2(canonical_technique_1_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_2_pdf, canonical_technique_3_canonical_reservoir_2_pdf, mis_weight_normalization);

                    if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_reservoir, random_number_generator))
                    {
                        selected_point_on_light = point_on_light;
                        selected_light_source_normal = light_source_normal;
                        selected_light_source_area = light_source_area;
                        selected_emission = emission;
                    }
                }
            }
        }

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
        /*float bsdf_sample_pdf;
        float3 sampled_bsdf_direction;
        BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

        BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
        ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

        bool intersection_found = false;
        BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;*/
        if (canonical_technique_3_canonical_reservoir_3_pdf > 0.0f)
        {
            /*hiprtRay new_ray;
            new_ray.origin = shading_point;
            new_ray.direction = sampled_bsdf_direction;

            intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index, ray_payload.bounce, random_number_generator);*/

            // Checking that we did hit something and if we hit something,
            // it needs to be emissive
            // if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black())
            {
                /*LightSampleInformation light_sample;
                light_sample.emission = shadow_light_ray_hit_info.hit_emission;
                light_sample.emissive_triangle_index = shadow_light_ray_hit_info.hit_prim_index;
                light_sample.light_area = triangle_area(render_data, shadow_light_ray_hit_info.hit_prim_index);
                light_sample.light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                light_sample.point_on_light = shading_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;*/

                // We want ReGIR to produce PDFs that are in area measure so we're converting from solid angle to area measure here
                // float area_measure_bsdf_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                /*float mis_weight = pairwise.compute_MIS_weight_for_BSDF_sample(render_data,
                    center_cell_surface, area_measure_bsdf_pdf, mis_weight_normalization,
                    non_canonical_RIS_integral_center_grid_cell, canonical_RIS_integral_center_grid_cell,

                    shading_point, view_direction, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index,
                    incident_light_info,

                    light_sample, random_number_generator);*/
                float mis_weight = pairwise.get_canonical_MIS_weight_3(canonical_technique_1_canonical_reservoir_3_pdf, canonical_technique_2_canonical_reservoir_3_pdf, canonical_technique_3_canonical_reservoir_3_pdf, mis_weight_normalization);

                float target_function = ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility,
                    ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility, false>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index,
                        ray_payload, canonical_technique_3_sample.point_on_light, canonical_technique_3_sample.light_source_normal, canonical_technique_3_sample.emission,
                        random_number_generator, canonical_technique_3_sample_ili);
                // Also converting the target function such that everything goes well when the target function is divided by the
                // source PDF in the reservoir.stream_sample() procedure.
                // 
                // Without the target function conversion, we're going to run into issues because if dividing only by the area measure
                // PDF, we're not going to recover the "true" BSDF contribution if it is not in area measure itself
                target_function = solid_angle_to_area_pdf(target_function, hippt::length(canonical_technique_3_sample.point_on_light - shading_point), compute_cosine_term_at_light_source(canonical_technique_3_sample.light_source_normal, -hippt::normalize(canonical_technique_3_sample.point_on_light - shading_point)));

                if (out_reservoir.stream_sample(mis_weight, target_function, canonical_technique_3_canonical_reservoir_3_pdf, canonical_technique_3_sample, random_number_generator))
                {
                    selected_point_on_light = canonical_technique_3_sample.point_on_light;
                    selected_light_source_normal = canonical_technique_3_sample.light_source_normal;
                    selected_light_source_area = canonical_technique_3_sample.light_area;
                    selected_emission = canonical_technique_3_sample.emission;
                    selected_incident_light_info = canonical_technique_3_sample_ili;
                }
            }
        }

        /*pairwise.sum_BSDF_sample_to_canonical_weights(render_data,
            canonical_technique_1_reservoir, canonical_technique_2_reservoir,
            canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_1_canonical_reservoir_2_pdf,
            canonical_technique_2_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_2_pdf,
            mis_weight_normalization,

            view_direction, shading_point, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index, random_number_generator);*/
#endif

        if (out_reservoir.weight_sum == 0.0f || out_need_fallback_sampling)
            return LightSampleInformation();

        out_reservoir.finalize_resampling(1.0f, 1.0f);
    }
#elif ReGIR_ShadingResamplingDoMISBalanceHeuristic == KERNEL_OPTION_TRUE
    {
        for (int neighbor = 0; neighbor < render_data.render_settings.regir_settings.shading.number_of_neighbors; neighbor++)
        {
            unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<false>(
                shading_point, render_data.current_camera, ray_payload.material.roughness,
                render_data.render_settings.regir_settings.shading.do_cell_jittering,
                render_data.render_settings.regir_settings.shading.jittering_radius, neighbor_rng);
            if (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                // Couldn't find a valid neighbor
                continue;
            else
                out_need_fallback_sampling = false;

            for (int i = 0; i < render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor; i++)
            {
                // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
                ReGIRReservoir non_canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(neighbor_grid_cell_index, neighbor_rng);

                if (non_canonical_reservoir.UCW <= 0.0f)
                    // No valid sample in that reservoir
                    continue;

                // TODO we evaluate the BSDF in there and then we're going to evaluate the BSDF again in the light sampling routine, that's double BSDF :(
                /*float3 point_on_light;
                float3 light_source_normal;
                float light_source_area;
                Xorshift32Generator rng_point_on_triangle(non_canonical_reservoir.sample.random_seed);
                if (!sample_point_on_generic_triangle(non_canonical_reservoir.sample.emissive_triangle_index, render_data.buffers.vertices_positions, render_data.buffers.triangles_indices,
                    rng_point_on_triangle,
                    point_on_light, light_source_normal, light_source_area))
                    continue;*/

                float3 point_on_light = non_canonical_reservoir.sample.point_on_light;
                float3 light_source_normal = get_triangle_normal_not_normalized(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
                float light_source_area = hippt::length(light_source_normal) * 0.5f;
                light_source_normal /= light_source_area * 2.0f;

                ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
                float target_function = ReGIR_shading_evaluate_target_function<
                    ReGIR_ShadingResamplingTargetFunctionVisibility,
                    ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal,
                        last_hit_primitive_index, ray_payload,
                        point_on_light, light_source_normal,
                        emission, random_number_generator);

                float mis_weight;
                {
                    float RIS_integral = render_data.render_settings.regir_settings.non_canonical_pre_integration_factors[neighbor_grid_cell_index];
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;
                    float non_canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_grid_cell_index, emission, light_source_normal, point_on_light, random_number_generator);
                    float non_canonical_sample_PDF = non_canonical_sample_PDF_unnormalized / RIS_integral;

                    float RIS_integral_canonical = render_data.render_settings.regir_settings.canonical_pre_integration_factors[neighbor_grid_cell_index];
                    if (RIS_integral_canonical == 0.0f)
                        RIS_integral_canonical = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral_canonical = 1.0f;

                    float canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, neighbor_grid_cell_index, emission, light_source_normal, point_on_light, random_number_generator);
                    float canonical_sample_PDF = canonical_sample_PDF_unnormalized / RIS_integral_canonical;
                    bool need_canonical_PDF = (ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_GridFillTargetFunctionCosineTerm || ReGIR_GridFillTargetFunctionCosineTermLightSource) && render_data.render_settings.regir_settings.DEBUG_INCLUDE_CANONICAL;
                    need_canonical_PDF |= render_data.render_settings.regir_settings.DEBUG_FORCE_REGIR8CANONICAL;
                    if (!need_canonical_PDF)
						canonical_sample_PDF = 0.0f;

                    BSDFIncidentLightInfo no_info = BSDFIncidentLightInfo::NO_INFO;
                    float BSDF_pdf;
                    BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, hippt::normalize(point_on_light - shading_point), no_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
                    ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, BSDF_pdf, random_number_generator);

                    BSDF_pdf = solid_angle_to_area_pdf(BSDF_pdf, hippt::length(point_on_light - shading_point), compute_cosine_term_at_light_source(light_source_normal, -hippt::normalize(point_on_light - shading_point)));
                    if constexpr (ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_FALSE)
						BSDF_pdf = 0.0f;

                    mis_weight = non_canonical_sample_PDF / (non_canonical_sample_PDF * render_data.render_settings.regir_settings.shading.number_of_neighbors * render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor + BSDF_pdf + canonical_sample_PDF);
                }

                if (out_reservoir.stream_reservoir(mis_weight, target_function, non_canonical_reservoir, random_number_generator))
                {
                    selected_point_on_light = point_on_light;
                    selected_light_source_normal = light_source_normal;
                    selected_light_source_area = light_source_area;
                    selected_emission = emission;
                }
            }
        }

        out_need_fallback_sampling = false;

        float bsdf_sample_pdf;
        float3 sampled_bsdf_direction;
        BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
        BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
        ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

        bool intersection_found = false;
        BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
        if (bsdf_sample_pdf > 0.0f)
        {
            hiprtRay new_ray;
            new_ray.origin = shading_point;
            new_ray.direction = sampled_bsdf_direction;

            intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index, ray_payload.bounce, random_number_generator);

            // Checking that we did hit something and if we hit something,
            // it needs to be emissive
            if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black())
            {
                LightSampleInformation light_sample;
                light_sample.emission = shadow_light_ray_hit_info.hit_emission;
                light_sample.emissive_triangle_index = shadow_light_ray_hit_info.hit_prim_index;
                light_sample.light_area = triangle_area(render_data, shadow_light_ray_hit_info.hit_prim_index);
                light_sample.light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                light_sample.point_on_light = shading_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;

                //float mis_weight = 1.0f;
                float mis_weight;
                {
                    unsigned int center_grid_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(shading_point, render_data.current_camera, ray_payload.material.roughness);
                    if (center_grid_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                    {
                        out_need_fallback_sampling = true;

                        return LightSampleInformation();
                    }

                    float RIS_integral = render_data.render_settings.regir_settings.non_canonical_pre_integration_factors[center_grid_index];
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;
                    float non_canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, center_grid_index, light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, random_number_generator);
                    float non_canonical_sample_PDF = non_canonical_sample_PDF_unnormalized / RIS_integral;

                    float RIS_integral_canonical = render_data.render_settings.regir_settings.canonical_pre_integration_factors[center_grid_index];
                    if (RIS_integral_canonical == 0.0f)
                        RIS_integral_canonical = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral_canonical = 1.0f;

                    float canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, center_grid_index, light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, random_number_generator);
                    float canonical_sample_PDF = canonical_sample_PDF_unnormalized / RIS_integral_canonical;
                    bool need_canonical_PDF = (ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_GridFillTargetFunctionCosineTerm || ReGIR_GridFillTargetFunctionCosineTermLightSource) && render_data.render_settings.regir_settings.DEBUG_INCLUDE_CANONICAL;
                    need_canonical_PDF |= render_data.render_settings.regir_settings.DEBUG_FORCE_REGIR8CANONICAL;
                    if (!need_canonical_PDF)
                        canonical_sample_PDF = 0.0f;
                    
                    float area_measure_bsdf_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                    
                    mis_weight = area_measure_bsdf_pdf / (non_canonical_sample_PDF * render_data.render_settings.regir_settings.shading.number_of_neighbors * render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor + area_measure_bsdf_pdf + canonical_sample_PDF);
                }

                float target_function = ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility,
                    ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility, false>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index,
                        ray_payload, light_sample.point_on_light, light_sample.light_source_normal, light_sample.emission,
                        random_number_generator, incident_light_info);

                // We want ReGIR to produce PDFs that are in area measure so we're converting from solid angle to area measure here
                float area_measure_bsdf_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                // Also converting the target function such that everything goes well when the target function is divided by the
                // source PDF in the reservoir.stream_sample() procedure.
                // 
                // Without the target function conversion, we're going to run into issues because if dividing only by the area measure
                // PDF, we're not going to recover the "true" BSDF contribution
                target_function = solid_angle_to_area_pdf(target_function, hippt::length(light_sample.point_on_light - shading_point), compute_cosine_term_at_light_source(light_sample.light_source_normal, -hippt::normalize(light_sample.point_on_light - shading_point)));

                if (out_reservoir.stream_sample(mis_weight, target_function, area_measure_bsdf_pdf, light_sample, random_number_generator))
                {
                    selected_point_on_light = light_sample.point_on_light;
                    selected_light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                    selected_light_source_area = light_sample.light_area;
                    selected_emission = shadow_light_ray_hit_info.hit_emission;
                    selected_incident_light_info = incident_light_info;
                }
            }
        }
#endif

        // Incorporating a canonical candidate if doing visibility reuse because visibility reuse
        // may cause the grid cell to produce no valid reservoir at all so we need canonical samples to
        // cover those cases for unbiased results
        bool need_canonical = (ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_GridFillTargetFunctionCosineTerm || ReGIR_GridFillTargetFunctionCosineTermLightSource) && render_data.render_settings.regir_settings.DEBUG_INCLUDE_CANONICAL;
        need_canonical |= render_data.render_settings.regir_settings.DEBUG_FORCE_REGIR8CANONICAL;
        if (need_canonical)
        {
            // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
            unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<true>(
                shading_point, render_data.current_camera, ray_payload.material.roughness,
                render_data.render_settings.regir_settings.shading.do_cell_jittering,
                render_data.render_settings.regir_settings.shading.jittering_radius, neighbor_rng);

            // Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
            // that cannot be resolved
            if (neighbor_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
            {
                // We found at least one good sample so we're not going to need a fallback on another light sampling strategy than ReGIR
                out_need_fallback_sampling = false;

                ReGIRReservoir canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<true>(neighbor_grid_cell_index, neighbor_rng);

                if (canonical_reservoir.UCW > 0.0f && canonical_reservoir.UCW != ReGIRReservoir::UNDEFINED_UCW)
                {
                    /*float3 point_on_light;
                    float3 light_source_normal;
                    float light_source_area;*/

                    ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, canonical_reservoir.sample.emissive_triangle_index);
                    /*Xorshift32Generator rng_point_on_triangle(canonical_reservoir.sample.random_seed);
                    if (sample_point_on_generic_triangle(canonical_reservoir.sample.emissive_triangle_index, render_data.buffers.vertices_positions, render_data.buffers.triangles_indices,
                        rng_point_on_triangle, point_on_light, light_source_normal, light_source_area))*/

                    float3 point_on_light = canonical_reservoir.sample.point_on_light;
                    float3 light_source_normal = get_triangle_normal_not_normalized(render_data, canonical_reservoir.sample.emissive_triangle_index);
                    float light_source_area = hippt::length(light_source_normal) * 0.5f;
                    light_source_normal /= light_source_area * 2.0f;

                    {
                        // Adding visibility in the canonical sample target function's if we have visibility reuse
                        // (or visibility in the grid fill target function) because otherwise this canonical sample
                        // will kill all the benefits of the visibility reuse
                        //
                        // TLDR is that this is pretty much necessary for good visibility reuse quality
                        float target_function = ReGIR_shading_evaluate_target_function<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                            shading_point, view_direction, shading_normal, geometric_normal,
                            last_hit_primitive_index, ray_payload,
                            point_on_light, light_source_normal,
                            emission, random_number_generator);

                        float mis_weight;
                        {
                            float RIS_integral = render_data.render_settings.regir_settings.non_canonical_pre_integration_factors[neighbor_grid_cell_index];
                            if (RIS_integral == 0.0f)
                                RIS_integral = 1.0f;
                            if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                                RIS_integral = 1.0f;
                            float non_canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_grid_cell_index, emission, light_source_normal, point_on_light, random_number_generator);
                            float non_canonical_sample_PDF = non_canonical_sample_PDF_unnormalized / RIS_integral;

                            float RIS_integral_canonical = render_data.render_settings.regir_settings.canonical_pre_integration_factors[neighbor_grid_cell_index];
                            if (RIS_integral_canonical == 0.0f)
                                RIS_integral_canonical = 1.0f;
                            if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                                RIS_integral_canonical = 1.0f;

                            float canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, neighbor_grid_cell_index, emission, light_source_normal, point_on_light, random_number_generator);
                            float canonical_sample_PDF = canonical_sample_PDF_unnormalized / RIS_integral_canonical;
                            bool need_canonical_PDF = (ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_GridFillTargetFunctionCosineTerm || ReGIR_GridFillTargetFunctionCosineTermLightSource) && render_data.render_settings.regir_settings.DEBUG_INCLUDE_CANONICAL;
                            need_canonical_PDF |= render_data.render_settings.regir_settings.DEBUG_FORCE_REGIR8CANONICAL;
                            if (!need_canonical_PDF)
                                canonical_sample_PDF = 0.0f;

                            BSDFIncidentLightInfo no_info = BSDFIncidentLightInfo::NO_INFO;
                            float BSDF_pdf;
                            BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, hippt::normalize(point_on_light - shading_point), no_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
                            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, BSDF_pdf, random_number_generator);

                            BSDF_pdf = solid_angle_to_area_pdf(BSDF_pdf, hippt::length(point_on_light - shading_point), compute_cosine_term_at_light_source(light_source_normal, -hippt::normalize(point_on_light - shading_point)));
                            if constexpr (ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_FALSE)
                                BSDF_pdf = 0.0f;

                            mis_weight = canonical_sample_PDF / (non_canonical_sample_PDF * render_data.render_settings.regir_settings.shading.number_of_neighbors * render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor + BSDF_pdf + canonical_sample_PDF);
                        }

                        if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_reservoir, random_number_generator))
                        {
                            selected_point_on_light = point_on_light;
                            selected_light_source_normal = light_source_normal;
                            selected_light_source_area = light_source_area;
                            selected_emission = emission;
                        }
                    }
                }
            }
        }

        if (out_reservoir.weight_sum == 0.0f || out_need_fallback_sampling)
            return LightSampleInformation();

        out_reservoir.finalize_resampling(1.0f, 1.0f);
    }
#else
    {
        int selected_neighbor = -1;

        for (int neighbor = 0; neighbor < render_data.render_settings.regir_settings.shading.number_of_neighbors; neighbor++)
        {
            unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<false>(
                shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
                render_data.render_settings.regir_settings.shading.get_do_cell_jittering(ray_payload.bounce == 0),
                render_data.render_settings.regir_settings.shading.jittering_radius, neighbor_rng);
            if (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                // Couldn't find a valid neighbor
                continue;
            else
                out_need_fallback_sampling = false;

            for (int i = 0; i < render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor; i++)
            {
                // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
                ReGIRReservoir non_canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(neighbor_grid_cell_index, ray_payload.bounce == 0, neighbor_rng);

                if (non_canonical_reservoir.UCW <= 0.0f)
                    // No valid sample in that reservoir
                    continue;

                // TODO we evaluate the BSDF in there and then we're going to evaluate the BSDF again in the light sampling routine, that's double BSDF :(
                /*float3 point_on_light;
                float3 light_source_normal;
                float light_source_area;
                Xorshift32Generator rng_point_on_triangle(non_canonical_reservoir.sample.random_seed);
                if (!sample_point_on_generic_triangle(non_canonical_reservoir.sample.emissive_triangle_index, render_data.buffers.vertices_positions, render_data.buffers.triangles_indices,
                    rng_point_on_triangle,
                    point_on_light, light_source_normal, light_source_area))
                    continue;*/

                float3 point_on_light = non_canonical_reservoir.sample.point_on_light;
                float3 light_source_normal = get_triangle_normal_not_normalized(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
                float light_source_area = hippt::length(light_source_normal) * 0.5f;
                light_source_normal /= light_source_area * 2.0f;

                ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
                float target_function = ReGIR_shading_evaluate_target_function<
                    ReGIR_ShadingResamplingTargetFunctionVisibility,
                    ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal,
                        last_hit_primitive_index, ray_payload,
                        point_on_light, light_source_normal,
                        emission, random_number_generator);

                float mis_weight = 1.0f;

                if (out_reservoir.stream_reservoir(mis_weight, target_function, non_canonical_reservoir, random_number_generator))
                {
                    selected_neighbor = neighbor;

                    selected_point_on_light = point_on_light;
                    selected_light_source_normal = light_source_normal;
                    selected_light_source_area = light_source_area;
                    selected_emission = emission;
                }
            }
        }

        out_need_fallback_sampling = false;

        float bsdf_sample_pdf;
        float3 sampled_bsdf_direction;
        BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
        BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
        ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

        bool intersection_found = false;
        BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
        if (bsdf_sample_pdf > 0.0f)
        {
            hiprtRay new_ray;
            new_ray.origin = shading_point;
            new_ray.direction = sampled_bsdf_direction;

            intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index, ray_payload.bounce, random_number_generator);

            // Checking that we did hit something and if we hit something,
            // it needs to be emissive
            if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black())
            {
                LightSampleInformation light_sample;
                light_sample.emission = shadow_light_ray_hit_info.hit_emission;
                light_sample.emissive_triangle_index = shadow_light_ray_hit_info.hit_prim_index;
                light_sample.light_area = triangle_area(render_data, shadow_light_ray_hit_info.hit_prim_index);
                light_sample.light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                light_sample.point_on_light = shading_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;

                float mis_weight = 1.0f;
                float target_function = ReGIR_shading_evaluate_target_function<
                    ReGIR_ShadingResamplingTargetFunctionVisibility,
                    ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility, false>(render_data,
                        shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index,
                        ray_payload, light_sample.point_on_light, light_sample.light_source_normal, light_sample.emission,
                        random_number_generator, incident_light_info);

                target_function = solid_angle_to_area_pdf(target_function, hippt::length(light_sample.point_on_light - shading_point), compute_cosine_term_at_light_source(light_sample.light_source_normal, -hippt::normalize(light_sample.point_on_light - shading_point)));

                float area_measure_bsdf_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                if (out_reservoir.stream_sample(mis_weight, target_function, area_measure_bsdf_pdf, light_sample, random_number_generator))
                {
                    selected_neighbor = render_data.render_settings.regir_settings.shading.number_of_neighbors;

                    selected_point_on_light = light_sample.point_on_light;
                    selected_light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                    selected_light_source_area = light_sample.light_area;
                    selected_emission = shadow_light_ray_hit_info.hit_emission;
                    selected_incident_light_info = incident_light_info;
                }
            }
        }
#endif

        // Incorporating a canonical candidate if doing visibility reuse because visibility reuse
        // may cause the grid cell to produce no valid reservoir at all so we need canonical samples to
        // cover those cases for unbiased results
        bool need_canonical = (ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_GridFillTargetFunctionCosineTerm || ReGIR_GridFillTargetFunctionCosineTermLightSource) && render_data.render_settings.regir_settings.DEBUG_INCLUDE_CANONICAL;
        need_canonical |= render_data.render_settings.regir_settings.DEBUG_FORCE_REGIR8CANONICAL;
        if (need_canonical)
        {
            // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
            unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<true>(
                shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
                false, // render_data.render_settings.regir_settings.shading.do_cell_jittering,
                render_data.render_settings.regir_settings.shading.jittering_radius, neighbor_rng);

            // Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
            // that cannot be resolved
            if (neighbor_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
            {
                // We found at least one good sample so we're not going to need a fallback on another light sampling strategy than ReGIR
                out_need_fallback_sampling = false;

                ReGIRReservoir canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<true>(neighbor_grid_cell_index, ray_payload.bounce == 0, neighbor_rng);

                if (canonical_reservoir.UCW > 0.0f && canonical_reservoir.UCW != ReGIRReservoir::UNDEFINED_UCW)
                {
                    /*float3 point_on_light;
                    float3 light_source_normal;
                    float light_source_area;*/

                    ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, canonical_reservoir.sample.emissive_triangle_index);
                    /*Xorshift32Generator rng_point_on_triangle(canonical_reservoir.sample.random_seed);
                    if (sample_point_on_generic_triangle(canonical_reservoir.sample.emissive_triangle_index, render_data.buffers.vertices_positions, render_data.buffers.triangles_indices,
                        rng_point_on_triangle, point_on_light, light_source_normal, light_source_area))*/

                    float3 point_on_light = canonical_reservoir.sample.point_on_light;
                    float3 light_source_normal = get_triangle_normal_not_normalized(render_data, canonical_reservoir.sample.emissive_triangle_index);
                    float light_source_area = hippt::length(light_source_normal) * 0.5f;
                    light_source_normal /= light_source_area * 2.0f;

                    {
                        // Adding visibility in the canonical sample target function's if we have visibility reuse
                        // (or visibility in the grid fill target function) because otherwise this canonical sample
                        // will kill all the benefits of the visibility reuse
                        //
                        // TLDR is that this is pretty much necessary for good visibility reuse quality
                        float target_function = ReGIR_shading_evaluate_target_function<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
                            shading_point, view_direction, shading_normal, geometric_normal,
                            last_hit_primitive_index, ray_payload,
                            point_on_light, light_source_normal,
                            emission, random_number_generator);

                        float mis_weight = 1.0f;
                        if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_reservoir, random_number_generator))
                        {
                            selected_neighbor = render_data.render_settings.regir_settings.shading.number_of_neighbors + ReGIR_ShadingResamplingDoBSDFMIS;

                            selected_point_on_light = point_on_light;
                            selected_light_source_normal = light_source_normal;
                            selected_light_source_area = light_source_area;
                            selected_emission = emission;
                        }
                    }
                }
            }
        }

        if (out_reservoir.weight_sum == 0.0f || out_need_fallback_sampling)
            return LightSampleInformation();

        neighbor_rng.m_state.seed = neighbor_rng_seed;

        float normalization_denominator = 0.0f;
        float normalization_numerator = 0.0f;
        for (int i = 0; i < render_data.render_settings.regir_settings.shading.number_of_neighbors + need_canonical + ReGIR_ShadingResamplingDoBSDFMIS; i++)
        {
            bool is_bsdf_sample = (i == render_data.render_settings.regir_settings.shading.number_of_neighbors && ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE);
            bool is_canonical = i == render_data.render_settings.regir_settings.shading.number_of_neighbors + ReGIR_ShadingResamplingDoBSDFMIS && need_canonical;

            unsigned int neighbor_cell_index;
            if (is_bsdf_sample)
            {
                neighbor_cell_index = render_data.render_settings.regir_settings.get_neighbor_replay_hash_grid_cell_index_for_shading(
                    shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
                    false,
                    false,
                    render_data.render_settings.regir_settings.shading.jittering_radius,
                    neighbor_rng);
            }
            else
            {
                neighbor_cell_index = render_data.render_settings.regir_settings.get_neighbor_replay_hash_grid_cell_index_for_shading(
                    shading_point, shading_normal, render_data.current_camera, ray_payload.material.roughness, ray_payload.bounce == 0,
                    is_canonical,
                    render_data.render_settings.regir_settings.shading.get_do_cell_jittering(ray_payload.bounce == 0),
                    render_data.render_settings.regir_settings.shading.jittering_radius,
                    neighbor_rng);
            }

            if (neighbor_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                // Outside of the alive grid
                //
                // Note that this also applies for the canonical sample because canonical samples are gathered
                // from neighbors. But if the neighbor is outside of the grid (or in a non-alive grid cell), then
                // we have no canonical neighbor to count in the MIS weights
                continue;

            if (selected_neighbor == i)
            {
                if (is_canonical)
                {
                    float RIS_integral = render_data.render_settings.regir_settings.get_canonical_pre_integration_factor(neighbor_cell_index, ray_payload.bounce == 0);
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;

                    float light_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, neighbor_cell_index, ray_payload.bounce == 0,
                        selected_emission, selected_light_source_normal, selected_point_on_light, random_number_generator);
                    float light_sample_PDF = light_sample_PDF_unnormalized / RIS_integral;

                    normalization_numerator = light_sample_PDF;
                    normalization_denominator += light_sample_PDF;
                }
                else if (is_bsdf_sample)
                {
                    float bsdf_pdf;
                    BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, hippt::normalize(out_reservoir.sample.point_on_light - shading_point), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
                    ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

                    bsdf_pdf = solid_angle_to_area_pdf(bsdf_pdf, hippt::length(out_reservoir.sample.point_on_light - shading_point), compute_cosine_term_at_light_source(selected_light_source_normal, -hippt::normalize(out_reservoir.sample.point_on_light - shading_point)));

                    normalization_numerator = bsdf_pdf;
                    normalization_denominator += bsdf_pdf;
                }
                else
                {
                    // Non-canonical sample

                    float RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(neighbor_cell_index, ray_payload.bounce == 0);
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;

                    float light_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_cell_index, ray_payload.bounce == 0,
                        selected_emission, selected_light_source_normal, selected_point_on_light, random_number_generator);
                    float light_sample_PDF = light_sample_PDF_unnormalized / RIS_integral;

                    normalization_numerator = light_sample_PDF;
                    normalization_denominator += light_sample_PDF * render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor;
                }

                continue;
            }
            else
            {
                if (is_canonical)
                {
                    float RIS_integral = render_data.render_settings.regir_settings.get_canonical_pre_integration_factor(neighbor_cell_index, ray_payload.bounce == 0);
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;

                    float light_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, neighbor_cell_index, ray_payload.bounce == 0,
                        selected_emission, selected_light_source_normal, selected_point_on_light, random_number_generator);
                    float light_sample_PDF = light_sample_PDF_unnormalized / RIS_integral;

                    normalization_denominator += light_sample_PDF;
                }
                else if (is_bsdf_sample)
                {
                    float bsdf_pdf;
                    BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, hippt::normalize(out_reservoir.sample.point_on_light - shading_point), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
                    ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

                    bsdf_pdf = solid_angle_to_area_pdf(bsdf_pdf, hippt::length(out_reservoir.sample.point_on_light - shading_point), compute_cosine_term_at_light_source(selected_light_source_normal, -hippt::normalize(out_reservoir.sample.point_on_light - shading_point)));

                    normalization_denominator += bsdf_pdf;
                }
                else
                {
                    // Non-canonical sample

                    float RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(neighbor_cell_index, ray_payload.bounce == 0);
                    if (RIS_integral == 0.0f)
                        RIS_integral = 1.0f;
                    if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                        RIS_integral = 1.0f;

                    float light_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_cell_index, ray_payload.bounce == 0, 
                        selected_emission, selected_light_source_normal, selected_point_on_light, random_number_generator);
                    float light_sample_PDF = light_sample_PDF_unnormalized / RIS_integral;

                    normalization_denominator += light_sample_PDF * render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor;
                }
            }
        }

        out_reservoir.finalize_resampling(normalization_numerator, normalization_denominator);
    }
#endif

    LightSampleInformation out_sample;

    // The UCW is the inverse of the PDF but we expect the PDF to be in 'area_measure_pdf', not the inverse PDF (UCW), so we invert it
    out_sample.area_measure_pdf = 1.0f / out_reservoir.UCW;
    out_sample.emissive_triangle_index = out_reservoir.sample.emissive_triangle_index;
    out_sample.emission = selected_emission;
    out_sample.light_area = selected_light_source_area;
    out_sample.light_source_normal = selected_light_source_normal;
    out_sample.point_on_light = selected_point_on_light;
    out_sample.incident_light_info = selected_incident_light_info;

    return out_sample;
}

template <int samplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle(const HIPRTRenderData& render_data, 
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, 
    int last_hit_primitive_index, RayPayload& ray_payload,
    Xorshift32Generator& random_number_generator)
{
    if constexpr (samplingStrategy == LSS_BASE_UNIFORM)
    {
        return sample_one_emissive_triangle_uniform(render_data, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_POWER)
    {
        return sample_one_emissive_triangle_power(render_data, random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_REGIR)
    {
        bool point_outside_grid = false;

        LightSampleInformation light_sample = sample_one_emissive_triangle_regir(render_data,
            shading_point, view_direction, shading_normal, geometric_normal, 
            last_hit_primitive_index, ray_payload,
            point_outside_grid,
            random_number_generator);

        if (!point_outside_grid)
            return light_sample;
        else
        {
#if ReGIR_FallbackLightSamplingStrategy == LSS_BASE_REGIR
            // Invalid fallback strategy
            invalid ReGIR light sampling fallback strategy
#endif

            // Fallback method as the point was outside of the ReGIR grid
            return sample_one_emissive_triangle<ReGIR_FallbackLightSamplingStrategy>(render_data,
                shading_point, view_direction, shading_normal, geometric_normal, 
                last_hit_primitive_index, ray_payload,
                random_number_generator);
        }
    }
}

/**
 * Overload of the function used when sampling lights without a world shading point (as in ReSTIR DI light presampling for example)
 * 
 * This means that positional light sampling schemes such as ReGIR or light trees cannot be used as the template argument here
 * and will produced incorrect results if used anyways
 */
template <int samplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator)
{
    RayPayload dummy_ray_payload;

    return sample_one_emissive_triangle<samplingStrategy>(render_data,
        make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f),
        -1, dummy_ray_payload,
        random_number_generator);
}

/**
 * 'clamp_condition' is an additional condition that needs to be met
 * for clamping to occur. If the additional condition is not met (the boolean
 * 'clamp_condition' is false, then the 'light_contribution' parameter is returned
 * untouched
 */
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F clamp_light_contribution(ColorRGB32F light_contribution, float clamp_max_value, bool clamp_condition)
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
 * This is the function that should always be called when you want the light PDF of a *light* sample for use in
 * MIS balance heurisitic (or power or whatever)
 * 
 * This function exists because some light samplers cannot be "point-evaluated" for a given BSDF sample. 
 * This is the case for ReGIR for example where we cannot compute the PDF of a given sample.
 * 
 * For these light samplers, the light PDF of a BSDF sample is approximated. 
 * This means that in a MIS weighting scheme, the light PDF of the BSDF sample is going to be approximated.
 * 
 * For MIS to stay correct, we also need to use that approximated light-PDF when computing the MIS weight in
 * the light sampling part of MIS, i.e. we cannot:
 * 
 * - Use the correct light PDF for computing the light sample MIS weight
 * - Use the approxmiated light PDF for computing the BSDF sample MIS weight
 * 
 * We need the approximated PDF in both places
 * 
 * This function computes the approximated PDF that should be used in the computation of all MIS weights (in the balance/power/... heuristic)
 * 
 * If the current light sampler is able to evaluate the correct light-PDF of a given BSDF sample, then this function will just return
 * 'original_pdf'. This is because is these cases, 'original_pdf' is already the perfect PDF so we don't to approximate anything
 * 
 * This function will do some computations for the approximated PDF only if the light sampler's PDF cannot be
 * evaluated for any given sample (case of ReGIR for example)
 */
template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE float light_sample_pdf_for_MIS_solid_angle_measure(const HIPRTRenderData& render_data,
    float original_pdf,
    float light_area,
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction,

    float roughness = 0,
	float3 shading_point = make_float3(0, 0, 0),
    float3 view_direction = make_float3(0, 0, 0),
	float3 surface_shading_normal = make_float3(0, 0, 0),
	float3 surface_geometric_normal = make_float3(0, 0, 0),
    int primitive_index = -1,
    float3 point_on_light = make_float3(0, 0, 0), Xorshift32Generator random_number_generator = Xorshift32Generator(42))
{
    // TODO this whole function should not be needed anymore
    return original_pdf;

    //if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
    //{
    //    // Approximating the ReGIR light PDF for the given BSDF sample with the basic NEE PDF
    //    RayPayload ray_payload;
    //    ray_payload.material.roughness = roughness;
    //    float target_function = ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(render_data,
    //        shading_point, view_direction, surface_shading_normal, surface_geometric_normal,
    //        primitive_index, ray_payload,
    //        point_on_light, light_surface_normal,
    //        light_emission, random_number_generator);

    //    unsigned int hash_grid_cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(shading_point, surface_shading_normal, render_data.current_camera, roughness);
    //    if (hash_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
    //    {
    //        float normalization = render_data.render_settings.regir_settings.non_canonical_pre_integration_factors[hash_grid_cell_index];
    //        if (normalization == 0.0f)
    //            // ReGIR cannot find a single good light sample for that cell so the integral is 0, no direct lighting
    //            return pdf_of_emissive_triangle_hit_solid_angle<LSS_BASE_REGIR>(render_data, light_area, light_emission, light_surface_normal, hit_distance, to_light_direction);
    //        
    //        return target_function / normalization;
    //    }
    //    else
    //    {
    //        float fake_ReGIR_PDF = pdf_of_emissive_triangle_hit_solid_angle<LSS_BASE_REGIR>(render_data, light_area, light_emission, light_surface_normal, hit_distance, to_light_direction);

    //        // Multipliying by an arbitrary factor since ReGIR is supposed to produce better light samples
    //        // This basically mimics the effect of resampling
    //        //
    //        // This is very arbitrary. Clamping at 100.0f. Very arbitrary
    //        fake_ReGIR_PDF *= hippt::min(100.0f, render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor * sqrtf(render_data.render_settings.regir_settings.grid_fill.light_sample_count_per_cell_reservoir) * sqrtf(render_data.render_settings.regir_settings.spatial_reuse.spatial_neighbor_count));

    //        return fake_ReGIR_PDF;
    //    }
    //}
    //else
    //    // If the light sampler does support the evaluation of the PDF, just returning the PDF unchanged
    //    // because this is already the exact PDF
    //    return original_pdf;
}

/**
 * Returns true if the given contribution satisfies the minimum light contribution
 * required for a light to be 
 */
HIPRT_DEVICE HIPRT_INLINE bool check_minimum_light_contribution(float minimum_contribution, const ColorRGB32F& contribution)
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

HIPRT_DEVICE HIPRT_INLINE bool check_minimum_light_contribution(float minimum_contribution, float contribution)
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
