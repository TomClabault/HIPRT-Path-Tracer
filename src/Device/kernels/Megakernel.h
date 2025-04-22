/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_MEGAKERNEL_H
#define KERNELS_MEGAKERNEL_H

#include "Device/includes/AdaptiveSampling.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/Lights.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Material.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) MegaKernel(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline MegaKernel(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    if (!render_data.aux_buffers.pixel_active[pixel_index])
        return;

    if (render_data.render_settings.do_render_low_resolution())
        // Reducing the number of bounces to 3 if rendering at low resolution
        // for better interactivity
        render_data.render_settings.nb_bounces = hippt::min(3, render_data.render_settings.nb_bounces);

#if ViewportColorOverriden == 1
    // If some kernel option is going to debug some color in the viewport,
    // then we're clearing the viewport buffer here
    render_data.buffers.accumulated_ray_colors[pixel_index] = ColorRGB32F();
#endif


    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator random_number_generator(seed);

    // Initializing the closest hit info the information from the camera ray pass
    HitInfo closest_hit_info;
    closest_hit_info.inter_point = render_data.g_buffer.primary_hit_position[pixel_index];
    closest_hit_info.geometric_normal = hippt::normalize(render_data.g_buffer.geometric_normals[pixel_index].unpack());
    closest_hit_info.shading_normal = hippt::normalize(render_data.g_buffer.shading_normals[pixel_index].unpack());
    closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];

    // Initializing the ray with the information from the camera ray pass
    hiprtRay ray;
    ray.direction = hippt::normalize(-render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index));

    RayPayload ray_payload;
    ray_payload.next_ray_state = RayState::BOUNCE;
    ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();

    // Because this is the camera hit (and assuming the camera isn't inside volumes for now),
    // the ray volume state after the camera hit is just an empty interior stack but with
    // the material index that we hit pushed onto the stack. That's it. Because it is that
    // simple, we don't have the ray volume state in the GBuffer but rather we can
    // reconstruct the ray volume state on the fly
    ray_payload.volume_state.reconstruct_first_hit(
        ray_payload.material,
        render_data.buffers.material_indices,
        closest_hit_info.primitive_index,
        random_number_generator);

    // This structure is going to contain the information for reusing the
    // BSDF ray when doing NEE with MIS: as a matter of fact, when doing
    // NEE with MIS, we're shooting a BSDF ray. If that ray doesn't hit
    // an emissive triangle, we can just reuse that ray for our indirect
    // bounce ray. That structure contains all that is necessary to reuse
    // the ray. This structure is filled by the emissive light sampling
    // or the envmap sampling function
    MISBSDFRayReuse mis_reuse;
    // + 1 to nb_bounces here because we want "0" bounces to still act as one
    // hit and to return some color
    bool intersection_found = closest_hit_info.primitive_index != -1;

    for (int& bounce = ray_payload.bounce; bounce < render_data.render_settings.nb_bounces + 1; bounce++)
    {
        if (ray_payload.next_ray_state != RayState::MISSED)
        {
            if (bounce > 0)
                intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, mis_reuse, random_number_generator);

            if (intersection_found)
            {
                if (bounce == 0)
                    store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);

                // TODO REMOVE THE DEBUG IF
                if (bounce > 0 || render_data.render_settings.enable_direct)
                    ray_payload.ray_color += estimate_direct_lighting(render_data, ray_payload, closest_hit_info, -ray.direction, x, y, mis_reuse, random_number_generator);

                BSDFIncidentLightInfo sampled_light_info; // This variable is never used, this is just for debugging on the CPU so that we know what the BSDF sampled
                bool valid_indirect_bounce = path_tracing_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray, mis_reuse, random_number_generator, &sampled_light_info);
                if (!valid_indirect_bounce)
                    // Bad BSDF sample (under the surface), killed by russian roulette, ...
                    break;
            }
            else
            {
                ray_payload.ray_color += path_tracing_miss_gather_envmap(render_data, ray_payload, ray.direction, pixel_index);
                ray_payload.next_ray_state = RayState::MISSED;
            }
        }
        else if (ray_payload.next_ray_state == RayState::MISSED)
            break;
    }

    // Checking for NaNs / negative value samples. Output 
    if (!sanity_check(render_data, ray_payload.ray_color, x, y))
        return;

    // If we got here, this means that we still have at least one ray active
    // This is a concurrent write by the way but we don't really care, everyone is writing
    // the same value
    render_data.aux_buffers.still_one_ray_active[0] = 1;

#if ViewportColorOverriden == 1
    // Modifying the ray color such that we display some debug color to the screen


#if DirectLightNEEPlusPlusDisplayShadowRaysDiscarded == KERNEL_OPTION_TRUE
    // Nothing to do, the debug is already handled in the shadow ray NEE function
#elif ReGIR_DebugMode != REGIR_DEBUG_MODE_NO_DEBUG
#if ReGIR_DebugMode  == REGIR_DEBUG_MODE_GRID_CELLS
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        // We have a first hit
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
        float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
        float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

        ray_payload.ray_color = render_data.render_settings.regir_settings.get_random_cell_color(primary_hit) * (render_data.render_settings.sample_number + 1);
        ray_payload.ray_color *= hippt::dot(shading_normal, view_direction);
    }
#elif ReGIR_DebugMode == REGIR_DEBUG_MODE_AVERAGE_CELL_NON_CANONICAL_RESERVOIR_CONTRIBUTION
    if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
    {
        float3 primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

        int cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(primary_hit);

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

        int cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(primary_hit);

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
        int cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(primary_hit);

        ColorRGB32F color;
        float3 rep_point = ReGIR_get_cell_representative_point(render_data, cell_index);

        if (rep_point.x != ReGIRRepresentative::UNDEFINED_POINT.x)
        {
            // Interpreting debug_view_scale_factor as a distance
            if (hippt::length(rep_point - primary_hit) < render_data.render_settings.regir_settings.debug_view_scale_factor)
                color = ColorRGB32F::random_color(cell_index + 1);
        }

        // Scaling by SPP so that the visualization doesn't get darker and darker with increasing number of SPP
        color *= render_data.render_settings.sample_number + 1;

        ray_payload.ray_color = ColorRGB32F(color);
    }
#endif
#endif
#endif

    path_tracing_accumulate_color(render_data, ray_payload.ray_color, pixel_index);
}

#endif
