/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_UTILS_H
#define DEVICE_LIGHT_UTILS_H

#include "Device/includes/PDFConversion.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE ColorRGB32F get_emission_of_triangle_from_index(HIPRTRenderData& render_data, int triangle_index)
{
    return render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[triangle_index]);
}

/**
 * Reference: [A Low-Distortion Map Between Triangle and Square, Heitz, 2019]
 * 
 * Maps a point in a square to a point in an arbitrary triangle
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void square_to_triangle(float& x, float& y)
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
}
/**
 * Samples a point uniformly on the given triangle (given with the triangle index)
 * 
 * Returns true if the sampling was successful, false otherwise (can fail if the triangle is way too small or degenerate)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool sample_point_on_triangle(int triangle_index, const float3* vertices_positions, const int* triangles_indices, Xorshift32Generator& rng,
    float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area)
{
    float3 vertex_A = vertices_positions[triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = vertices_positions[triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = vertices_positions[triangles_indices[triangle_index * 3 + 2]];

    float rand_1 = rng();
    float rand_2 = rng();

#if TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_TURK_1990
    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;
#elif TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_HEITZ_2019
    square_to_triangle(rand_1, rand_2);

    float u = rand_1;
    float v = rand_2;
#endif

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;
    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;

    float3 normal = hippt::cross(AB, AC);
    float length_normal = hippt::length(normal);
    if (length_normal <= 1.0e-6f)
        return false;

    out_sample_point = random_point_on_triangle;
    out_sampled_triangle_normal = normal / length_normal;
    out_triangle_area = 0.5f * length_normal;

    return true;
}

/**
 * The PDF is computed in area measure
 */
HIPRT_HOST_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_uniform(
    const int* triangles_indices, const int* emissive_triangles_indices, int emissive_triangle_count,
    const float3* vertices_positions,
    const int* material_indices, const DevicePackedTexturedMaterialSoA& materials_buffer,
    Xorshift32Generator& random_number_generator)
{
    LightSampleInformation light_sample;

    int random_index = random_number_generator.random_index(emissive_triangle_count);
    int triangle_index = emissive_triangles_indices[random_index];

    float sampled_triangle_area;
    float3 sampled_triangle_normal;
    float3 random_point_on_triangle;
    if (!sample_point_on_triangle(triangle_index, vertices_positions, triangles_indices, random_number_generator, random_point_on_triangle, sampled_triangle_normal, sampled_triangle_area))
        return LightSampleInformation();

    light_sample.emissive_triangle_index = triangle_index;
    light_sample.light_source_normal = sampled_triangle_normal;
    light_sample.light_area = sampled_triangle_area;
    light_sample.emission = materials_buffer.get_emission(material_indices[triangle_index]);
    light_sample.point_on_light = random_point_on_triangle;

    // PDF of that point on that triangle
    light_sample.area_measure_pdf = 1.0f / sampled_triangle_area;
    // PDF of that triangle sampled uniformly amongst all emissive triangles
    light_sample.area_measure_pdf /= emissive_triangle_count;

    return light_sample;
}

HIPRT_HOST_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_power_area(
    const DeviceAliasTable& power_area_alias_table,
    const int* triangles_indices, const int* emissive_triangles_indices,
    const float3* vertices_positions,
    const int* material_indices, const DevicePackedTexturedMaterialSoA& materials_buffer,
    Xorshift32Generator& random_number_generator)
{
    LightSampleInformation out_sample;

    int random_index = power_area_alias_table.sample(random_number_generator);
    int triangle_index = emissive_triangles_indices[random_index];

    float sampled_triangle_area;
    float3 sampled_triangle_normal;
    float3 random_point_on_triangle;
    if (!sample_point_on_triangle(triangle_index, vertices_positions, triangles_indices, random_number_generator, random_point_on_triangle, sampled_triangle_normal, sampled_triangle_area))
        return LightSampleInformation();

    out_sample.emissive_triangle_index = triangle_index;
    out_sample.light_source_normal = sampled_triangle_normal;
    out_sample.light_area = sampled_triangle_area;
    out_sample.emission = materials_buffer.get_emission(material_indices[triangle_index]);
    out_sample.point_on_light = random_point_on_triangle;

    // PDF of that point on that triangle
    out_sample.area_measure_pdf = 1.0f / sampled_triangle_area;
    // PDF of sampling that triangle according to its luminance-area
    out_sample.area_measure_pdf *= (out_sample.emission.luminance() * sampled_triangle_area) / power_area_alias_table.sum_elements;

    return out_sample;
}

HIPRT_HOST_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_regir(
    const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, 
    int last_hit_primitive_index, RayPayload& ray_payload,
    bool& shading_point_outside_of_grid,
    Xorshift32Generator& random_number_generator)
{
    LightSampleInformation out_sample;
    ReGIRReservoir out_reservoir;
    ColorRGB32F picked_sample_emission;

    for (int i = 0; i < render_data.render_settings.regir_settings.shading.cell_reservoir_resample_per_shading_point; i++)
    {
        ReGIRReservoir cell_reservoir = render_data.render_settings.regir_settings.get_cell_reservoir_for_shading(shading_point, shading_point_outside_of_grid, random_number_generator, render_data.render_settings.regir_settings.shading.do_cell_jittering);
        if (shading_point_outside_of_grid)
            continue;
        else if (cell_reservoir.UCW == 0.0f)
            // No valid sample in that reservoir
            continue;

        ColorRGB32F current_emission = render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[cell_reservoir.sample.emissive_triangle_index]);

        float mis_weight = 1.0f / render_data.render_settings.regir_settings.shading.cell_reservoir_resample_per_shading_point;
        float target_function = ReGIR_shading_evaluate_target_function(render_data, 
            shading_point, view_direction, shading_normal, geometric_normal, 
            last_hit_primitive_index, ray_payload,
            cell_reservoir, current_emission,
            random_number_generator);

        if (out_reservoir.stream_reservoir(mis_weight, target_function, cell_reservoir, random_number_generator))
            picked_sample_emission = current_emission;
    }

    if (out_reservoir.weight_sum == 0.0f)
        return LightSampleInformation();

    out_reservoir.finalize_resampling();

    // The UCW is the inverse of the PDF but we expect the PDF to be in 'area_measure_pdf', not the inverse PDF, so we invert it
    out_sample.area_measure_pdf = 1.0f / out_reservoir.UCW;
    out_sample.emission = picked_sample_emission;
    out_sample.emissive_triangle_index = out_reservoir.sample.emissive_triangle_index;
    out_sample.light_area = out_reservoir.sample.light_area;
    out_sample.light_source_normal = out_reservoir.sample.light_source_normal.unpack();
    out_sample.point_on_light = out_reservoir.sample.point_on_light;

    return out_sample;
}

template <int samplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_HOST_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle(const HIPRTRenderData& render_data, 
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, 
    int last_hit_primitive_index, RayPayload& ray_payload,
    Xorshift32Generator& random_number_generator)
{
    if constexpr (samplingStrategy == LSS_BASE_UNIFORM)
    {
        return sample_one_emissive_triangle_uniform(render_data.buffers.triangles_indices, render_data.buffers.emissive_triangles_indices, render_data.buffers.emissive_triangles_count,
            render_data.buffers.vertices_positions,
            render_data.buffers.material_indices, render_data.buffers.materials_buffer,
            random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_POWER_AREA)
    {
        return sample_one_emissive_triangle_power_area(render_data.buffers.emissives_power_area_alias_table, render_data.buffers.triangles_indices, render_data.buffers.emissive_triangles_indices,
            render_data.buffers.vertices_positions,
            render_data.buffers.material_indices, render_data.buffers.materials_buffer,
            random_number_generator);
    }
    else if constexpr (samplingStrategy == LSS_BASE_REGIR)
    {
        bool point_outside_grid;

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
HIPRT_HOST_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator)
{
    RayPayload dummy_ray_payload;

    return sample_one_emissive_triangle<samplingStrategy>(render_data,
        make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f),
        -1, dummy_ray_payload,
        random_number_generator);
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
HIPRT_HOST_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, int hit_primitive_index, ColorRGB32F light_emission)
{
#if DirectLightSamplingBaseStrategy == LSS_BASE_UNIFORM || (DirectLightSamplingBaseStrategy == LSS_BASE_REGIR && ReGIR_GridFillLightSamplingBaseStrategy == LSS_BASE_UNIFORM)
    // Surface area PDF of hitting that point on that triangle in the scene
    float light_area = triangle_area(render_data, hit_primitive_index);
    float area_measure_pdf = 1.0f / light_area;
    area_measure_pdf /= render_data.buffers.emissive_triangles_count;
#elif DirectLightSamplingBaseStrategy == LSS_BASE_POWER_AREA || (DirectLightSamplingBaseStrategy == LSS_BASE_REGIR && ReGIR_GridFillLightSamplingBaseStrategy == LSS_BASE_POWER_AREA)
    // Note that for ReGIR, we cannot have the exact light PDF since ReGIR is based on RIS so we're
    // faking it with power-area PDF

    float light_area = triangle_area(render_data, hit_primitive_index);
    float area_measure_pdf = 1.0f / light_area;
    area_measure_pdf *= (light_emission.luminance() * light_area) / render_data.buffers.emissives_power_area_alias_table.sum_elements;
#endif
     
    return area_measure_pdf;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, const ShadowLightRayHitInfo& light_hit_info)
{
    return pdf_of_emissive_triangle_hit_area_measure(render_data, light_hit_info.hit_prim_index, light_hit_info.hit_emission);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, int hit_primitive_index, 
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    // abs() here to allow backfacing lights
    // Without abs() here:
    //  - We could be hitting the back of an emissive triangle (think of quad light hanging in the air)
    //  --> triangle normal not facing the same way 
    //  --> cos_angle negative
    float cosine_light_source = hippt::abs(hippt::dot(light_surface_normal, to_light_direction));

    float pdf_area_measure = pdf_of_emissive_triangle_hit_area_measure(render_data, hit_primitive_index, light_emission);

    return area_to_solid_angle_pdf(pdf_area_measure, hit_distance, cosine_light_source);
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
HIPRT_HOST_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, const ShadowLightRayHitInfo& light_hit_info, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle(render_data, 
        light_hit_info.hit_prim_index, light_hit_info.hit_emission, light_hit_info.hit_shading_normal, 
        light_hit_info.hit_distance, to_light_direction);
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
HIPRT_HOST_DEVICE HIPRT_INLINE float light_sample_pdf_for_MIS_solid_angle_measure(const HIPRTRenderData& render_data,
    float original_pdf,
    int hit_primitive_index,
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
#if DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
    // Approximating the ReGIR light PDF for the given BSDF sample with the basic NEE PDF
    return pdf_of_emissive_triangle_hit_solid_angle(render_data, hit_primitive_index, light_emission, light_surface_normal, hit_distance, to_light_direction);
#else
    // If the light sampler does support the evaluation of the PDF, just returning the PDF unchanged
    // because this is the exact PDF
    return original_pdf;
#endif
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
