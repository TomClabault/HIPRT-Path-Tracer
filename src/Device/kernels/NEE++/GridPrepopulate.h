/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NEE_PLUS_PLUS_GRID_PREPOPULATE_H
#define KERNELS_NEE_PLUS_PLUS_GRID_PREPOPULATE_H

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

HIPRT_DEVICE void accumulate_NEE_plus_plus(HIPRTRenderData& render_data, const hiprtRay& ray, const HitInfo& closest_hit_info, RayPayload& ray_payload, Xorshift32Generator& random_number_generator)
{
    // Just making sure that this is not set to false
    render_data.nee_plus_plus.m_update_visibility_map = true;

    for (int sample = 0; sample < 100; sample++)
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle<LSS_BASE_POWER>(render_data,
            closest_hit_info.inter_point, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
            closest_hit_info.primitive_index, ray_payload,
            random_number_generator);

        if (light_sample.area_measure_pdf <= 0.0f)
            // Can happen for very small triangles
            continue;

        float3 shadow_ray_origin = closest_hit_info.inter_point;
        float3 shadow_ray_direction = light_sample.point_on_light - shadow_ray_origin;
        float distance_to_light = hippt::length(shadow_ray_direction);
        float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

        hiprtRay shadow_ray;
        shadow_ray.origin = shadow_ray_origin;
        shadow_ray.direction = shadow_ray_direction_normalized;

        ColorRGB32F light_source_radiance;
        // abs() here to allow backfacing light sources
        float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);
        if (dot_light_source > 0.0f)
        {
            NEEPlusPlusContext nee_plus_plus_context;
            nee_plus_plus_context.point_on_light = light_sample.point_on_light;
            nee_plus_plus_context.shaded_point = shadow_ray_origin;
            bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator, ray_payload.bounce);
        }
    }
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) NEEPlusPlus_Grid_Prepopulate(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline NEEPlusPlus_Grid_Prepopulate(HIPRTRenderData render_data, int x, int y)
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

    hiprtRay ray = render_data.current_camera.get_camera_ray(x_ray_point_direction, y_ray_point_direction, render_data.render_settings.render_resolution);
    RayPayload ray_payload;

    HitInfo closest_hit_info;
    bool intersection_found = trace_main_path_ray(render_data, ray, ray_payload, closest_hit_info, /* camera ray = no previous primitive hit */ -1, /* bounce. Always 0 for camera rays*/ 0, random_number_generator);

    if (!intersection_found)
        return;

    MISBSDFRayReuse mis_reuse;
    for (int& bounce = ray_payload.bounce; bounce < render_data.render_settings.nb_bounces + 1; bounce++)
    {
        if (ray_payload.next_ray_state != RayState::MISSED)
        {
            if (bounce > 0)
                intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, mis_reuse, random_number_generator);

            if (intersection_found)
            {
                accumulate_NEE_plus_plus(render_data, ray, closest_hit_info, ray_payload, random_number_generator);

                BSDFIncidentLightInfo sampled_light_info; // This variable is never used, this is just for debugging on the CPU so that we know what the BSDF sampled
                bool valid_indirect_bounce = path_tracing_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray, mis_reuse, random_number_generator, &sampled_light_info);
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
