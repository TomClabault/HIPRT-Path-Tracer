/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_VISIBILITY_REUSE_H
#define DEVICE_KERNELS_REGIR_VISIBILITY_REUSE_H

#include "Device/includes/Intersect.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * Returns false if the given 'point_on_light' is occluded from the point of view of the
 * representative point of the grid cell given by 'linear_cell_index'
 * 
 * Returns true if unoccluded
 */
HIPRT_HOST_DEVICE bool ReGIR_grid_cell_visibility_test(const HIPRTRenderData& render_data, int linear_cell_index, float3 point_on_light, Xorshift32Generator& rng)
{
    int pixel_index_for_representative_point = render_data.render_settings.regir_settings.get_representative_point_index(linear_cell_index);

    float3 shadow_ray_origin;
    if (pixel_index_for_representative_point == -1)
        // No representative point yet, using the center of the cell
        shadow_ray_origin = render_data.render_settings.regir_settings.get_cell_center_from_linear(linear_cell_index);
    else
        shadow_ray_origin = render_data.g_buffer.primary_hit_position[pixel_index_for_representative_point];

    float3 to_light_direction = point_on_light - shadow_ray_origin;
    float distance_to_light = hippt::length(to_light_direction);
    to_light_direction /= distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = shadow_ray_origin;
    shadow_ray.direction = to_light_direction;

    return !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light, -1, 0, rng);
}

HIPRT_HOST_DEVICE ReGIRReservoir visibility_reuse(const HIPRTRenderData& render_data, const ReGIRReservoir& current_reservoir,
    int linear_cell_index, Xorshift32Generator& rng)
{
    ReGIRReservoir out_reservoir = current_reservoir;

#if ReGIR_DoVisibilityReuse == KERNEL_OPTION_TRUE
    if (current_reservoir.UCW > 0.0f)
        if (!ReGIR_grid_cell_visibility_test(render_data, linear_cell_index, current_reservoir.sample.point_on_light, rng))
            out_reservoir.UCW = -1.0f;
#endif

    return out_reservoir;
}

#endif
