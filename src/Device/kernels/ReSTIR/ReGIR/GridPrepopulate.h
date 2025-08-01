/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_REGIR_GRID_PREPOPULATE_H
#define KERNELS_REGIR_GRID_PREPOPULATE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReGIR_Grid_Prepopulate(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Prepopulate(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * ReGIR_GridPrepopulationResolutionDownscale;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * ReGIR_GridPrepopulationResolutionDownscale;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

    Xorshift32Generator random_number_generator(seed);

    // Direction to the center of the pixel
    float x_ray_point_direction = (x + 0.5f);
    float y_ray_point_direction = (y + 0.5f);
    if (render_data.current_camera.do_jittering)
    {
        // Jitter randomly around the center
        x_ray_point_direction += random_number_generator() - 0.5f;
        y_ray_point_direction += random_number_generator() - 0.5f;
    }

    hiprtRay camera_ray = render_data.current_camera.get_camera_ray(x_ray_point_direction, y_ray_point_direction, render_data.render_settings.render_resolution);
    RayPayload ray_payload;

    HitInfo closest_hit_info;
    bool intersection_found = trace_main_path_ray(render_data, camera_ray, ray_payload, closest_hit_info, /* camera ray = no previous primitive hit */ -1, /* bounce. Always 0 for camera rays*/ 0, random_number_generator);

    if (!intersection_found)
        return;

    ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, closest_hit_info.shading_normal, render_data.current_camera, closest_hit_info.primitive_index, true, ray_payload.material);

	MISBSDFRayReuse mis_reuse;
    for (int& bounce = ray_payload.bounce; bounce < render_data.render_settings.nb_bounces + 1; bounce++)
    {
        if (ray_payload.next_ray_state != RayState::MISSED)
        {
            if (bounce > 0)
                intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, camera_ray, ray_payload, closest_hit_info, mis_reuse, random_number_generator);

            if (intersection_found)
            {
                if (bounce > 0)
                {
                    bool ReGIR_primary_hit = render_data.render_settings.regir_settings.compute_is_primary_hit(ray_payload);

                    // Storing data for ReGIR representative points
                    ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, closest_hit_info.shading_normal, render_data.current_camera, closest_hit_info.primitive_index, ReGIR_primary_hit, ray_payload.material);
                }

                BSDFIncidentLightInfo sampled_light_info; // This variable is never used, this is just for debugging on the CPU so that we know what the BSDF sampled
                bool valid_indirect_bounce = path_tracing_compute_next_indirect_bounce<true>(render_data, ray_payload, closest_hit_info, -camera_ray.direction, camera_ray, mis_reuse, random_number_generator, &sampled_light_info);
                if (!valid_indirect_bounce)
                    // Bad BSDF sample (under the surface), killed by russian roulette, ...
                    break;
            }
            else
                return;
        }
    }
}

#endif
