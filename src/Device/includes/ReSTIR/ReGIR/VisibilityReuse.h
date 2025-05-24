/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_VISIBILITY_REUSE_H
#define DEVICE_KERNELS_REGIR_VISIBILITY_REUSE_H

#include "Device/includes/Intersect.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"
#include "Device/includes/ReSTIR/ReGIR/VisibilityTest.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ReGIRReservoir visibility_reuse(const HIPRTRenderData& render_data, const ReGIRReservoir& current_reservoir,
    int hash_grid_cell_index, Xorshift32Generator& rng)
{
    ReGIRReservoir out_reservoir = current_reservoir;

#if ReGIR_DoVisibilityReuse == KERNEL_OPTION_TRUE
    if (current_reservoir.UCW > 0.0f)
    {
		Xorshift32Generator rng_point_on_triangle(current_reservoir.sample.random_seed);

        float3 point_on_light;
		float3 light_normal_trash;
		float triangle_area_trash;
        sample_point_on_generic_triangle(current_reservoir.sample.emissive_triangle_index,
            render_data.buffers.vertices_positions, render_data.buffers.triangles_indices, rng_point_on_triangle,
            point_on_light, light_normal_trash, triangle_area_trash);

        if (!ReGIR_grid_cell_visibility_test(render_data, hash_grid_cell_index, point_on_light, rng))
            out_reservoir.UCW = ReGIRReservoir::VISIBILITY_REUSE_KILLED_UCW;
    }
#endif

    return out_reservoir;
}

#endif
