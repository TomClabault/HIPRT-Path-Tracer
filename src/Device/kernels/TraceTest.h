/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_TRACE_TEST_H
#define KERNELS_TRACE_TEST_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"

#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) TraceTest(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline TraceTest(HIPRTRenderData render_data, int2 res, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= res.x || y >= res.y)
        return;

    uint32_t pixel_index = x + y * res.x;

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

    hiprtRay ray = render_data.current_camera.get_camera_ray(x_ray_point_direction, y_ray_point_direction, res);
    hiprtHit hit;

#ifdef __KERNELCC__
#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#if SharedStackBVHTraversalSize > 0
    hiprtSharedStackBuffer shared_stack_buffer{ SharedStackBVHTraversalSize, shared_stack_cache };
#else
    hiprtSharedStackBuffer shared_stack_buffer{ 0, nullptr };
#endif
    hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);

    hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> traversal(render_data.GPU_BVH, ray, global_stack, hiprtTraversalHintDefault);
#else
    hiprtGeomTraversalClosest traversal(render_data.GPU_BVH, ray, hiprtTraversalHintDefault);
#endif

    hit = traversal.getNextHit();
#endif

    render_data.g_buffer.first_hit_prim_index[pixel_index] = hit.hasHit() ? 1 : 0;
}

#endif
