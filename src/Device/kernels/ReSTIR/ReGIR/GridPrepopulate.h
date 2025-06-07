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
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReGIR_Grid_Prepopulate(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Prepopulate(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * ReGIR_GridPrepoluationResolutionDownscale;
    const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * ReGIR_GridPrepoluationResolutionDownscale;
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
    ray_payload.bounce = 0;
    ray_payload.next_ray_state = RayState::BOUNCE;

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
                if (bounce > 0)
                    // Storing data for ReGIR representative points
                    ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, render_data.current_camera, closest_hit_info.shading_normal, closest_hit_info.primitive_index, render_data.g_buffer.materials[pixel_index].unpack());

                BSDFIncidentLightInfo sampled_light_info; // This variable is never used, this is just for debugging on the CPU so that we know what the BSDF sampled
                bool valid_indirect_bounce = path_tracing_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray, mis_reuse, random_number_generator, &sampled_light_info);
                if (!valid_indirect_bounce)
                    // Bad BSDF sample (under the surface), killed by russian roulette, ...
                    break;
            }
            else
                ray_payload.next_ray_state = RayState::MISSED;
        }
        else if (ray_payload.next_ray_state == RayState::MISSED)
            break;
    }
}

#endif
