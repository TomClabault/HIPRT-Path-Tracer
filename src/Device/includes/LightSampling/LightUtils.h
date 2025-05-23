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

HIPRT_DEVICE ColorRGB32F get_emission_of_triangle_from_index(HIPRTRenderData& render_data, int triangle_index)
{
    return render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[triangle_index]);
}

/**
 * Reference: [A Low-Distortion Map Between Triangle and Square, Heitz, 2019]
 * 
 * Maps a point in a square to a point in an arbitrary triangle
 */
HIPRT_DEVICE HIPRT_INLINE float2 square_to_triangle(float x, float y)
{
    float2 remapped;

    if (y > x) 
    {
        remapped.x = x * 0.5f;
        remapped.y = y - x;
    }
    else 
    {
        remapped.y = y * 0.5f;
        remapped.x = x - y;
    }

    return remapped;
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
    float2 rands = square_to_triangle(rand_1, rand_2);

    float u = rands.x;
    float v = rands.y;
#endif

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;
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
    float2 rands = square_to_triangle(rand_1, rand_2);

    float u = rands.x;
    float v = rands.y;
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

HIPRT_DEVICE HIPRT_INLINE LightSampleInformation sample_one_emissive_triangle_regir(
    const HIPRTRenderData& render_data,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    int last_hit_primitive_index, RayPayload& ray_payload,
    bool& shading_point_outside_of_grid,
    Xorshift32Generator& random_number_generator)
{
    // Starting with this at true and if we find a single good neighbor,
    // this will be set o false
    shading_point_outside_of_grid = true;

    LightSampleInformation out_sample;
    ReGIRReservoir out_reservoir;

    // Some random seed to generate to positions of the neighbors (when jittering)
    // XORing here because not XORing was causing RNG correlations issues...
    // not sure how that works but more randomness here seems to be getting rid of those correlations issues
    unsigned neighbor_rng_seed = random_number_generator.xorshift32() ^ random_number_generator.xorshift32();
    Xorshift32Generator neighbor_rng(neighbor_rng_seed);
    int selected_neighbor = -1;

    for (int neighbor = 0; neighbor < render_data.render_settings.regir_settings.shading.number_of_neighbors; neighbor++)
    {
        unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<false>(shading_point, render_data.current_camera, neighbor_rng);
        if (neighbor_grid_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
            // Couldn't find a valid neighbor
            continue;
        else
            shading_point_outside_of_grid = false;

        for (int i = 0; i < render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor; i++)
        {
            // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
            ReGIRReservoir non_canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(neighbor_grid_cell_index, neighbor_rng);

            if (non_canonical_reservoir.UCW <= 0.0f)
                // No valid sample in that reservoir
                continue;

            // TODO we evaluate the BSDF in there and then we're going to evaluate the BSDF again in the light sampling routine, that's double BSDF :(
            float mis_weight = 1.0f;
            float target_function = ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility>(render_data, 
                shading_point, view_direction, shading_normal, geometric_normal, 
                last_hit_primitive_index, ray_payload, non_canonical_reservoir,
                random_number_generator);

            if (out_reservoir.stream_reservoir(mis_weight, target_function, non_canonical_reservoir, random_number_generator))
                selected_neighbor = neighbor;
        }
}

    // Incorporating a canonical candidate if doing visibility reuse because visibility reuse
    // may cause the grid cell to produce no valid reservoir at all so we need canonical samples to
    // cover those cases for unbiased results
    bool need_canonical = (ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility || ReGIR_GridFillTargetFunctionCosineTerm || ReGIR_GridFillTargetFunctionCosineTermLightSource) && render_data.render_settings.regir_settings.DEBUG_INCLUDE_CANONICAL;
    if (need_canonical)
    {
        // Will be set to true if the jittering causes the current shading point to be jittered out of the scene
        unsigned int neighbor_grid_cell_index = render_data.render_settings.regir_settings.find_valid_jittered_neighbor_cell_index<true>(shading_point, render_data.current_camera, neighbor_rng);

        // Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
        // that cannot be resolved
        if (neighbor_grid_cell_index != ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
        {
            ReGIRReservoir canonical_reservoir = render_data.render_settings.regir_settings.get_random_reservoir_in_grid_cell_for_shading<true>(neighbor_grid_cell_index, neighbor_rng);

            // Adding visibility in the canonical sample target function's if we have visibility reuse
            // (or visibility in the grid fill target function) because otherwise this canonical sample
            // will kill all the benefits of the visibility reuse
            //
            // This is pretty much necessary for good visibility reuse quality
            float target_function = ReGIR_shading_evaluate_target_function<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility>(render_data,
                shading_point, view_direction, shading_normal, geometric_normal,
                last_hit_primitive_index, ray_payload, canonical_reservoir,
                random_number_generator);

            float mis_weight = 1.0f;
            if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_reservoir, random_number_generator))
                selected_neighbor = render_data.render_settings.regir_settings.shading.number_of_neighbors;
        }
    }

    if (out_reservoir.weight_sum == 0.0f || shading_point_outside_of_grid)
        return LightSampleInformation();

    neighbor_rng.m_state.seed = neighbor_rng_seed;
    
    float normalization_weight = 0.0f;
    for (int i = 0; i < render_data.render_settings.regir_settings.shading.number_of_neighbors + need_canonical; i++)
    {
        bool is_canonical = i == render_data.render_settings.regir_settings.shading.number_of_neighbors;
            
		unsigned int neighbor_cell_index = render_data.render_settings.regir_settings.get_neighbor_replay_hash_grid_cell_index_for_shading(
            shading_point, render_data.current_camera,
            is_canonical, 
            neighbor_rng, render_data.render_settings.regir_settings.shading.do_cell_jittering);

        if (neighbor_cell_index == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
            // Outside of the alive grid
            //
            // Note that this also applies for the canonical sample because canonical samples are gathered
            // from neighbors. But if the neighbor is outside of the grid (or in a non-alive grid cell), then
            // we have no canonical neighbor to count in the MIS weights
            continue;

        if (is_canonical)
        {
            // Canonical reservoir.
            // This one can always produce any sample so this is always a valid neighbor
            normalization_weight += 1.0f;

            continue;
        }
        else if (selected_neighbor == i)
        {
            if (is_canonical)
                normalization_weight += 1.0f;
            else
                normalization_weight += render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor;

            continue;
        }

        if (ReGIR_shading_can_sample_be_produced_by(render_data, out_reservoir.sample, neighbor_cell_index, random_number_generator))
            normalization_weight += render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor;
    }

    out_reservoir.finalize_resampling(normalization_weight);

    // The UCW is the inverse of the PDF but we expect the PDF to be in 'area_measure_pdf', not the inverse PDF, so we invert it
    out_sample.area_measure_pdf = 1.0f / out_reservoir.UCW;
    out_sample.emissive_triangle_index = out_reservoir.sample.emissive_triangle_index;
    out_sample.emission = out_reservoir.sample.emission.unpack();
    out_sample.light_area = out_reservoir.sample.light_area;
    out_sample.light_source_normal = out_reservoir.sample.light_source_normal.unpack();
    out_sample.point_on_light = out_reservoir.sample.point_on_light;

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
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data, const ShadowLightRayHitInfo& light_hit_info)
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
HIPRT_DEVICE HIPRT_INLINE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data, const ShadowLightRayHitInfo& light_hit_info, float3 to_light_direction)
{
    return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data, 
        light_hit_info.hit_prim_index, light_hit_info.hit_emission, light_hit_info.hit_geometric_normal, 
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
template <int lightSamplingStrategy = DirectLightSamplingBaseStrategy>
HIPRT_DEVICE HIPRT_INLINE float light_sample_pdf_for_MIS_solid_angle_measure(const HIPRTRenderData& render_data,
    float original_pdf,
    float light_area,
    ColorRGB32F light_emission, float3 light_surface_normal,
    float hit_distance, float3 to_light_direction)
{
    if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
    {
        // Approximating the ReGIR light PDF for the given BSDF sample with the basic NEE PDF
        float fake_ReGIR_PDF = pdf_of_emissive_triangle_hit_solid_angle<LSS_BASE_REGIR>(render_data, light_area, light_emission, light_surface_normal, hit_distance, to_light_direction);

        // Multipliying by an arbitrary factor since ReGIR is supposed to produce better light samples
        // This basically mimics the effect of resampling
        //
        // This is very arbitrary. Clamping at 100.0f. Very arbitrary
        fake_ReGIR_PDF *= hippt::min(100.0f, render_data.render_settings.regir_settings.shading.reservoir_tap_count_per_neighbor * sqrtf(render_data.render_settings.regir_settings.grid_fill.sample_count_per_cell_reservoir) * sqrtf(render_data.render_settings.regir_settings.spatial_reuse.spatial_neighbor_count));

        return fake_ReGIR_PDF;
    }
    else
        // If the light sampler does support the evaluation of the PDF, just returning the PDF unchanged
        // because this is already the exact PDF
        return original_pdf;
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
