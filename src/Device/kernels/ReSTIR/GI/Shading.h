/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_SHADING_H
#define KERNELS_RESTIR_GI_SHADING_H

#include "Device/includes/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Lights.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_Shading(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_Shading(HIPRTRenderData render_data, int x, int y)
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

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator random_number_generator(seed);

    hiprtRay ray;
    ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    HitInfo closest_hit_info;
    closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];
    if (closest_hit_info.primitive_index == -1)
    {
        // Geometry miss, directly into the envmap
        ColorRGB32F envmap_radiance = path_tracing_miss_gather_envmap(render_data, 0, ColorRGB32F(1.0f), ray.direction, pixel_index);
        path_tracing_accumulate_color(render_data, envmap_radiance, pixel_index);

        return;
    }

    closest_hit_info.inter_point = render_data.g_buffer.primary_hit_position[pixel_index];
    closest_hit_info.shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();

    // Initializing the ray with the information from the camera ray pass

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


    float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
    
    ColorRGB32F camera_outgoing_radiance;
    ColorRGB32F DEBUG_COLOR;
    ColorRGB32F DEBUG_FIRST_HIT_THROUGHPUT;
    ColorRGB32F DEBUG_FIRST_BSDF_COLOR_DOT;

    ReSTIRGIReservoir resampling_reservoir = render_data.render_settings.restir_gi_settings.restir_output_reservoirs[pixel_index];

    if (render_data.render_settings.nb_bounces > 0)
    {
        // Only doing the ReSTIR GI stuff if we have more than 1 bounce

        if (!resampling_reservoir.sample.outgoing_radiance_to_visible_point.is_black())
        {
            // Only doing the shading if we do actually have a sample

            float3 shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
            float3 geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();

            float3 restir_resampled_indirect_direction;
            if (resampling_reservoir.sample.is_envmap_path())
                restir_resampled_indirect_direction = resampling_reservoir.sample.sample_point;
            else
                restir_resampled_indirect_direction = hippt::normalize(resampling_reservoir.sample.sample_point - closest_hit_info.inter_point);

            float bsdf_pdf;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, ray_payload.material, ray_payload.volume_state, false, view_direction, shading_normal, geometric_normal, restir_resampled_indirect_direction, bsdf_pdf, random_number_generator, 0, resampling_reservoir.sample.incident_light_info);
            if (bsdf_pdf > 0.0f)
            {
                // If we're here, it's supposedly because we do have a valid ReSTIR GI sample that is not null/black
                // But because of float imprecisions, the BSDF may still end re-evaluating to 0.0f for that sample
                // so we need that check here

                DEBUG_COLOR = ColorRGB32F(restir_resampled_indirect_direction);
                ColorRGB32F first_hit_throughput = bsdf_color * hippt::abs(hippt::dot(restir_resampled_indirect_direction, shading_normal)) * resampling_reservoir.UCW;
                DEBUG_FIRST_HIT_THROUGHPUT = first_hit_throughput;
                DEBUG_FIRST_BSDF_COLOR_DOT = bsdf_color * hippt::abs(hippt::dot(restir_resampled_indirect_direction, shading_normal));
                camera_outgoing_radiance += first_hit_throughput * resampling_reservoir.sample.outgoing_radiance_to_visible_point;
            }
        }
    }

    MISBSDFRayReuse mis_reuse;
    if (render_data.render_settings.enable_direct > 0)
        camera_outgoing_radiance += estimate_direct_lighting(render_data, ray_payload, closest_hit_info, view_direction, x, y, mis_reuse, random_number_generator);

    if (x == render_data.render_settings.debug_x && y == render_data.render_settings.debug_y)
#ifndef __KERNELCC__
        if (render_data.render_settings.sample_number % 2 == 0)
#else
        if (render_data.render_settings.sample_number > render_data.render_settings.stop_value)
#endif
        {
            for (int i = 0; i < 10; i++)
            {
                float value = hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[i], 0.0f);
                if (value != 0.0f)
                    printf("Average %d: %e\n", i + 1, value / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0));
            }

            printf("\n");
        }

    /*printf("Average 1: %e\nAverage 2: %e\nAverage 3: %e\nAverage 4: %e\nAverage 5: %e\nUCW: %e\n\n",
        hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[0], 0.0f) / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0),
        hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[1], 0.0f) / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0),
        hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[2], 0.0f) / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0),
        hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[3], 0.0f) / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0),
        hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[4], 0.0f) / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0),
        resampling_reservoir.UCW);*/

    ray_payload.ray_color = camera_outgoing_radiance;
    if (!sanity_check(render_data, ray_payload, x, y))
        return;

    if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::FINAL_RESERVOIR_UCW)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.UCW) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::TARGET_FUNCTION)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.sample.target_function) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::WEIGHT_SUM)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.weight_sum) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::M_COUNT)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.M) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else
        // Regular output
        path_tracing_accumulate_color(render_data, camera_outgoing_radiance, pixel_index);
}

#endif
