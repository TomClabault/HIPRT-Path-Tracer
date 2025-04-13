/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NEE_PLUS_PLUS_CACHE_PREPASS_H
#define KERNELS_NEE_PLUS_PLUS_CACHE_PREPASS_H

#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE hiprtHit simple_closest_hit(const HIPRTRenderData& render_data, hiprtRay ray, int last_primitive_index, Xorshift32Generator& random_number_generator)
{
    hiprtHit hit;

#ifdef __KERNELCC__
    // Payload for the alpha testing filter function
    FilterFunctionPayload payload;
    payload.render_data = &render_data;
    payload.random_number_generator = &random_number_generator;
    payload.last_hit_primitive_index = last_primitive_index;

#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#if SharedStackBVHTraversalSize > 0
    hiprtSharedStackBuffer shared_stack_buffer{ SharedStackBVHTraversalSize, shared_stack_cache };
#else
    hiprtSharedStackBuffer shared_stack_buffer{ 0, nullptr };
#endif
    hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);

    hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> traversal(render_data.GPU_BVH, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#else
    hiprtGeomTraversalClosest traversal(render_data.GPU_BVH, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#endif

    hit = traversal.getNextHit();
#else
hit = intersect_scene_cpu(render_data, ray, last_primitive_index, random_number_generator);
#endif

    return hit;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) NEEPlusPlusCachingPrepass(HIPRTRenderData render_data, unsigned int caching_sample_count)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline NEEPlusPlusCachingPrepass(HIPRTRenderData render_data, unsigned int caching_sample_count, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    unsigned int seed = wang_hash((pixel_index + 1) * render_data.random_number);
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

    // First finding where the camera ray intersects
    hiprtRay ray = render_data.current_camera.get_camera_ray(x_ray_point_direction, y_ray_point_direction, render_data.render_settings.render_resolution);
    hiprtHit hit = simple_closest_hit(render_data, ray, -1, random_number_generator);

    if (!hit.hasHit())
        return;

    // We have the intersection of the camera rays, we can already update the visibility map with those rays
    float3 intersection_position = ray.origin + ray.direction * hit.t;
    int camera_hit_primitive_index = hit.primID;

    render_data.nee_plus_plus.accumulate_visibility(NEEPlusPlusContext{ render_data.current_camera.position, intersection_position }, true);

    // Now sampling random lights from the camera ray first hit and caching the visibility
    for (int i = 0; i < caching_sample_count; i++)
    {
        float trash_pdf;
        LightSampleInformation light_info;

        float3 target_point = sample_one_emissive_triangle(render_data, random_number_generator, trash_pdf, light_info);
        float3 direction = target_point - intersection_position;
        float distance_to_point = hippt::length(direction);
        direction /= distance_to_point;

        hiprtRay shadow_ray;
        shadow_ray.origin = intersection_position;
        shadow_ray.direction = direction;

        hiprtHit shadow_ray_hit = simple_closest_hit(render_data, shadow_ray, camera_hit_primitive_index, random_number_generator);
        if (!shadow_ray_hit.hasHit())
            // Should never happen because we should at least hit the emissive triangle sampled
            continue;

        NEEPlusPlusContext context;
        context.shaded_point = intersection_position;
        context.point_on_light = target_point;

        // Is the point on the light visible?
        bool visible = shadow_ray_hit.hasHit() && shadow_ray_hit.primID == light_info.emissive_triangle_index;
        render_data.nee_plus_plus.accumulate_visibility(context, visible);
    }
}

#endif
