/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PATH_TRACING_H
#define DEVICE_INCLUDES_PATH_TRACING_H

#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightUtils.h"
#include "Device/includes/MISBSDFRayReuse.h"
#include "Device/includes/RussianRoulette.h"
#include "Device/includes/WarpDirectionReuse.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE bool path_tracing_find_indirect_bounce_intersection(HIPRTRenderData& render_data, hiprtRay ray, RayPayload& out_ray_payload, HitInfo& out_closest_hit_info, MISBSDFRayReuse mis_reuse, Xorshift32Generator& random_number_generator)
{
	if (mis_reuse.has_ray())
		// Reusing a BSDF MIS ray if there is one available
		return reuse_mis_ray(render_data, -ray.direction, out_ray_payload, out_closest_hit_info, mis_reuse);
	else
		// Not tracing for the primary ray because this has already been done in the camera ray pass
		return trace_main_path_ray(render_data, ray, out_ray_payload, out_closest_hit_info, out_closest_hit_info.primitive_index, out_ray_payload.bounce, random_number_generator);
}

HIPRT_DEVICE void path_tracing_sample_next_indirect_bounce(HIPRTRenderData& render_data, RayPayload& ray_payload, HitInfo& closest_hit_info, float3 view_direction, ColorRGB32F& out_bsdf_color, float3& out_bounce_direction, float& out_bsdf_pdf, MISBSDFRayReuse& mis_reuse, Xorshift32Generator& random_number_generator, BSDFIncidentLightInfo* out_sampled_light_info = nullptr)
{
    if (mis_reuse.has_ray())
        out_bsdf_color = reuse_mis_bsdf_sample(out_bounce_direction, out_bsdf_pdf, ray_payload, mis_reuse, out_sampled_light_info);
    else
    {
        BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f), *out_sampled_light_info, ray_payload.volume_state, true, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness);
        out_bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, out_bounce_direction, out_bsdf_pdf, random_number_generator);
    }

    ray_payload.accumulate_roughness(*out_sampled_light_info);

#if DoFirstBounceWarpDirectionReuse
    warp_direction_reuse(render_data, closest_hit_info, ray_payload, -ray.direction, bounce_direction, bsdf_color, bsdf_pdf, bounce, random_number_generator);
#endif
}

/**
 * Returns the new ray throughput after attenuation of the given 'current_throughput'
 */
HIPRT_DEVICE ColorRGB32F path_tracing_update_ray_throughput(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo& closest_hit_info, ColorRGB32F current_throughput, float& rr_throughput_scaling, ColorRGB32F bsdf_color, float3 bounce_direction, float bsdf_pdf, Xorshift32Generator& random_number_generator, bool apply_russian_roulette = true)
{
    ColorRGB32F throughput_attenuation = bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal)) / bsdf_pdf;
    // Russian roulette
    if (apply_russian_roulette && !do_russian_roulette(render_data.render_settings, ray_payload.bounce, current_throughput, rr_throughput_scaling, throughput_attenuation, random_number_generator))
        return ColorRGB32F(0.0f);

    // Dispersion ray throughput filter
    current_throughput *= get_dispersion_ray_color(ray_payload.volume_state.sampled_wavelength, ray_payload.material.dispersion_scale);
    current_throughput *= throughput_attenuation;
    // Clamp every component to a minimum of 1.0e-5f to avoid numerical instabilities that can
    // happen: with some material, the throughput can get so low that it becomes denormalized and
    // this can cause issues in some parts of the renderer (most notably the NaN detection)
    current_throughput.max(ColorRGB32F(1.0e-5f, 1.0e-5f, 1.0e-5f));

    ray_payload.next_ray_state = RayState::BOUNCE;

    return current_throughput;
}

/**
 * Returns the new ray throughput after attenuation of the given 'current_throughput'
 */
HIPRT_DEVICE ColorRGB32F path_tracing_update_ray_throughput(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo& closest_hit_info, ColorRGB32F current_throughput, ColorRGB32F bsdf_color, float3 bounce_direction, float bsdf_pdf, Xorshift32Generator& random_number_generator, bool apply_russian_roulette = true)
{
    float unused_rr_throughput_scaling;
    return path_tracing_update_ray_throughput(render_data, ray_payload, closest_hit_info, current_throughput, unused_rr_throughput_scaling, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator, apply_russian_roulette);
}

/**
 * Returns true if the bounce was sampled successfully,
 * false otherwise (is the BSDF sample failed, if russian roulette killed the sample, ...)
 */
HIPRT_DEVICE bool path_tracing_compute_next_indirect_bounce(HIPRTRenderData& render_data, RayPayload& ray_payload, HitInfo& closest_hit_info, float3 view_direction, hiprtRay& out_ray, MISBSDFRayReuse& mis_reuse, Xorshift32Generator& random_number_generator, BSDFIncidentLightInfo* incident_light_info = nullptr)
{
    ColorRGB32F bsdf_color;
    float3 bounce_direction;
    float bsdf_pdf;
    path_tracing_sample_next_indirect_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf, mis_reuse, random_number_generator, incident_light_info);

    // Terminate ray if bad sampling
    if (bsdf_pdf <= 0.0f)
        return false;

    ray_payload.throughput = path_tracing_update_ray_throughput(render_data, ray_payload, closest_hit_info, ray_payload.throughput, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator);
    if (ray_payload.throughput.is_black())
        // Killed by russian roulette
        return false;

    out_ray.origin = closest_hit_info.inter_point;
    out_ray.direction = bounce_direction;

    return true;
}

HIPRT_DEVICE void store_denoiser_AOVs(HIPRTRenderData& render_data, uint32_t pixel_index, float3 shading_normal, ColorRGB32F base_color)
{
    if (render_data.render_settings.sample_number == 0)
        render_data.aux_buffers.denoiser_albedo[pixel_index] = base_color;
    else
        render_data.aux_buffers.denoiser_albedo[pixel_index] = (render_data.aux_buffers.denoiser_albedo[pixel_index] * render_data.render_settings.denoiser_AOV_accumulation_counter + base_color) / (render_data.render_settings.denoiser_AOV_accumulation_counter + 1.0f);

    if (render_data.render_settings.sample_number == 0)
        render_data.aux_buffers.denoiser_normals[pixel_index] = shading_normal;
    else
    {
        float3 accumulated_normal = (render_data.aux_buffers.denoiser_normals[pixel_index] * render_data.render_settings.denoiser_AOV_accumulation_counter + shading_normal) / (render_data.render_settings.denoiser_AOV_accumulation_counter + 1.0f);
        float normal_length = hippt::length(accumulated_normal);
        if (!hippt::is_zero(normal_length))
            // Checking that it is non-zero otherwise we would accumulate a persistent NaN in the buffer when normalizing by the 0-length
            render_data.aux_buffers.denoiser_normals[pixel_index] = accumulated_normal / normal_length;
    }
}

HIPRT_DEVICE ColorRGB32F path_tracing_miss_gather_envmap(HIPRTRenderData& render_data, const ColorRGB32F& ray_throughput, float3 ray_direction, int bounce, uint32_t pixel_index)
{
    ColorRGB32F skysphere_color;

    if (render_data.world_settings.ambient_light_type == AmbientLightType::UNIFORM || render_data.bsdfs_data.white_furnace_mode)
        skysphere_color = render_data.world_settings.uniform_light_color;
    else if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP && render_data.world_settings.envmap_intensity == 0.0f)
        return ColorRGB32F(0.0f);
    else if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
    {
#if EnvmapSamplingStrategy != ESS_NO_SAMPLING
        // If we have sampling, only taking envmap into account on camera ray miss
        if (bounce == 0)
#endif
        {
            // We're only getting the skysphere radiance for the first rays because the
            // syksphere is importance sampled.
            skysphere_color = eval_envmap_no_pdf(render_data.world_settings, ray_direction);

#if EnvmapSamplingStrategy == ESS_NO_SAMPLING
            // If we don't have envmap sampling, we're only going to unscale on
            // bounce 0 (which is when a ray misses directly --> background color).
            // Otherwise, if not bounce 2, we do want to take the scaling into
            // account so this if will fail and the envmap color will never be unscaled
            if (!render_data.world_settings.envmap_scale_background_intensity && bounce == 0)
#else
            if (!render_data.world_settings.envmap_scale_background_intensity)
#endif
                // Un-scaling the envmap if the user doesn't want to scale the background
                skysphere_color /= render_data.world_settings.envmap_intensity;
        }
    }

    skysphere_color = clamp_light_contribution(skysphere_color, render_data.render_settings.envmap_contribution_clamp, /* clamp condition */ true);

    ColorRGB32F indirect_lighting_contribution = skysphere_color * ray_throughput;
    // Only clamping with the indirect lighting clamp value if
    // this is bounce > 0 (thanks to /* clamp condition */ bounce > 0)
    ColorRGB32F clamped_indirect_lighting_contribution = clamp_light_contribution(
        indirect_lighting_contribution, render_data.render_settings.indirect_contribution_clamp,
        /* clamp condition */ bounce > 0);

    if (bounce == 0)
        // The camera ray missed so we don't have the normals but we have the base color
        store_denoiser_AOVs(render_data, pixel_index, make_float3(0, 0, 0), skysphere_color);

    return clamped_indirect_lighting_contribution;
}

HIPRT_DEVICE ColorRGB32F path_tracing_miss_gather_envmap(HIPRTRenderData& render_data, RayPayload& ray_payload, float3 ray_direction, uint32_t pixel_index)
{
    return path_tracing_miss_gather_envmap(render_data, ray_payload.throughput, ray_direction, ray_payload.bounce, pixel_index);
}

HIPRT_DEVICE void path_tracing_accumulate_color(const HIPRTRenderData& render_data, const ColorRGB32F& ray_color, uint32_t pixel_index)
{
#if DisplayOnlySampleN == KERNEL_OPTION_TRUE
     int sampleIndex = render_data.render_settings.output_debug_sample_N;

     if (render_data.render_settings.sample_number == sampleIndex)
     {
         render_data.buffers.accumulated_ray_colors[pixel_index] = ray_color;
#if ViewportColorOverriden == 0
         render_data.buffers.accumulated_ray_colors[pixel_index] *= (sampleIndex + 1);
#endif
     }

     return;
#endif

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        float squared_luminance_of_samples = ray_color.luminance() * ray_color.luminance();
        // We can only use these buffers if the adaptive sampling or the stop noise threshold is enabled.
        // Otherwise, the buffers are destroyed to save some VRAM so they are not accessible
        render_data.aux_buffers.pixel_squared_luminance[pixel_index] += squared_luminance_of_samples;
    }

    if (render_data.render_settings.sample_number == 0)
        render_data.buffers.accumulated_ray_colors[pixel_index] = ray_color;
    else
        // If we are at a sample that is not 0, this means that we are accumulating
        render_data.buffers.accumulated_ray_colors[pixel_index] += ray_color;

    if (render_data.buffers.gmon_estimator.sets != nullptr)
    {
        // GMoN is in use, accumulating in the GMoN sets

        unsigned int offset = render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y * render_data.buffers.gmon_estimator.next_set_to_accumulate + pixel_index;

        if (render_data.render_settings.sample_number == 0)
            render_data.buffers.gmon_estimator.sets[offset] = ray_color;
        else
            render_data.buffers.gmon_estimator.sets[offset] += ray_color;
    }
}

HIPRT_DEVICE void path_tracing_accumulate_debug_view_color(const HIPRTRenderData& render_data, RayPayload& ray_payload, int pixel_index, Xorshift32Generator& rng)
{
#if ViewportColorOverriden == 1
    // Modifying the ray color such that we display some debug color to the screen


#if DirectLightNEEPlusPlusDisplayShadowRaysDiscarded == KERNEL_OPTION_TRUE
    // Nothing to do, the debug is already handled in the shadow ray NEE function
#elif NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        // We have a first hit
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
        float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
        float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
        
        unsigned int trash_checksum;
        NEEPlusPlusContext context;
        context.envmap = false;
        context.point_on_light = make_float3(0, 0, 0);
        context.shaded_point = primary_hit;
        context.shaded_point_surface_normal = shading_normal;

        ray_payload.ray_color = ColorRGB32F::random_color(render_data.nee_plus_plus.hash_context(context, render_data.current_camera, trash_checksum));
        ray_payload.ray_color *= (render_data.render_settings.sample_number + 1);
        ray_payload.ray_color *= hippt::dot(shading_normal, view_direction);
    }
#elif ReGIR_DebugMode != REGIR_DEBUG_MODE_NO_DEBUG
#if ReGIR_DebugMode  == REGIR_DEBUG_MODE_GRID_CELLS
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        // We have a first hit
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
        float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
        float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
        float primary_hit_roughness = render_data.g_buffer.materials[pixel_index].get_roughness();

        ray_payload.ray_color = render_data.render_settings.regir_settings.get_random_cell_color(primary_hit, shading_normal, render_data.current_camera, primary_hit_roughness, true);
        ray_payload.ray_color *= (render_data.render_settings.sample_number + 1);
        ray_payload.ray_color *= hippt::dot(shading_normal, view_direction);
    }
#elif ReGIR_DebugMode == REGIR_DEBUG_MODE_AVERAGE_CELL_NON_CANONICAL_RESERVOIR_CONTRIBUTION
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

        unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(primary_hit);

        float average_contribution = 0.0f;
        for (int i = 0; i < render_data.render_settings.regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell(); i++)
        {
            ReGIRReservoir reservoir = render_data.render_settings.regir_settings.get_cell_non_canonical_reservoir_from_cell_reservoir_index(cell_index, i);
            average_contribution += reservoir.sample.target_function * reservoir.UCW;
        }

        // Averaging
        average_contribution /= render_data.render_settings.regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell();
        // Scaling by the debug factor for visualization purposes
        average_contribution *= render_data.render_settings.regir_settings.debug_view_scale_factor;
        // Scaling by SPP
        average_contribution *= (render_data.render_settings.sample_number + 1);

        ray_payload.ray_color = ColorRGB32F(average_contribution);
    }
#elif ReGIR_DebugMode == REGIR_DEBUG_MODE_AVERAGE_CELL_CANONICAL_RESERVOIR_CONTRIBUTION
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

        unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(primary_hit);

        float average_contribution = 0.0f;
        for (int i = 0; i < render_data.render_settings.regir_settings.grid_fill.get_canonical_reservoir_count_per_cell(); i++)
        {
            ReGIRReservoir reservoir = render_data.render_settings.regir_settings.get_cell_canonical_reservoir_from_cell_reservoir_index(cell_index, i);
            average_contribution += reservoir.sample.target_function * reservoir.UCW;
        }

        // Averaging
        average_contribution /= render_data.render_settings.regir_settings.grid_fill.get_canonical_reservoir_count_per_cell();
        // Scaling by the debug factor for visualization purposes
        average_contribution *= render_data.render_settings.regir_settings.debug_view_scale_factor;
        // Scaling by SPP
        average_contribution *= (render_data.render_settings.sample_number + 1);

        ray_payload.ray_color = ColorRGB32F(average_contribution);
    }
#elif ReGIR_DebugMode == REGIR_DEBUG_MODE_REPRESENTATIVE_POINTS
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
        float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
        float primary_hit_roughness = render_data.g_buffer.materials[pixel_index].get_roughness();

        unsigned int cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(primary_hit, shading_normal, render_data.current_camera, primary_hit_roughness, true);

        ColorRGB32F color;
        float3 rep_point = ReGIR_get_cell_world_point(render_data, cell_index, true);
        // Interpreting debug_view_scale_factor as a distance
        if (hippt::length(rep_point - primary_hit) < render_data.render_settings.regir_settings.debug_view_scale_factor)
            color = ColorRGB32F::random_color(cell_index + 1);

        // Scaling by SPP so that the visualization doesn't get darker and darker with increasing number of SPP
        color *= render_data.render_settings.sample_number + 1;

        ray_payload.ray_color = ColorRGB32F(color);
    }
#elif ReGIR_DebugMode == REGIR_DEBUG_PRE_INTEGRATION_CHECK
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
        float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
        float3 geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
        int prim_index = render_data.g_buffer.first_hit_prim_index[pixel_index];
        float primary_hit_roughness = render_data.g_buffer.materials[pixel_index].get_roughness();
        unsigned int hash_grid_cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos(primary_hit, shading_normal, render_data.current_camera, primary_hit_roughness, true);
        if (hash_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
        {
            ray_payload.ray_color = ColorRGB32F(1.0f, 0.0f, 1.0f) * (render_data.render_settings.sample_number + 1);
            return;
        }

        RayPayload local_ray_payload;
        local_ray_payload.next_ray_state = RayState::BOUNCE;
        local_ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();

        // Because this is the camera hit (and assuming the camera isn't inside volumes for now),
        // the ray volume state after the camera hit is just an empty interior stack but with
        // the material index that we hit pushed onto the stack. That's it. Because it is that
        // simple, we don't have the ray volume state in the GBuffer but rather we can
        // reconstruct the ray volume state on the fly
        local_ray_payload.volume_state.reconstruct_first_hit(
            local_ray_payload.material,
            render_data.buffers.material_indices,
            prim_index,
            rng);

        ReGIRGridFillSurface surface = ReGIR_get_cell_surface(render_data, hash_grid_cell_index, true);

        float integral = 0.0f;
        constexpr unsigned int iterations = 500;
        for (int i = 0; i < iterations; i++)
        {
            LightSampleInformation light_sample = sample_one_emissive_triangle_power(render_data, rng);
            if (light_sample.area_measure_pdf <= 0.0f)
            {
                ray_payload.ray_color = ColorRGB32F(1.0f, 0.0f, 1.0f) * (render_data.render_settings.sample_number + 1);
                return;
            }

            float target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, surface, light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
            float RIS_integral = render_data.render_settings.regir_settings.get_non_canonical_pre_integration_factor(hash_grid_cell_index, true);
            if (RIS_integral == 0.0f)
                RIS_integral = 1.0f;
            if (!render_data.render_settings.regir_settings.DEBUG_DO_RIS_INTEGRAL_NORMALIZATION)
                RIS_integral = 1.0f;

            float PDF = target_function / RIS_integral;
            integral += PDF / iterations;
        }

        ColorRGB32F color;
        if (integral >= 0.0f && integral <= 1.0f)
            color = hippt::lerp(ColorRGB32F(1.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 1.0f, 0.0f), integral);
        else if (integral > 1.0f)
            color = hippt::lerp(ColorRGB32F(0.0f, 1.0f, 0.0f), ColorRGB32F(1.0f, 0.0f, 0.0f), hippt::min(1.0f, integral - 1.0f));

        // Scaling by SPP so that the visualization doesn't get darker and darker with increasing number of SPP
        color *= render_data.render_settings.sample_number + 1;
        ray_payload.ray_color = ColorRGB32F(color);
    }
#endif
#endif
#endif
}

#endif
