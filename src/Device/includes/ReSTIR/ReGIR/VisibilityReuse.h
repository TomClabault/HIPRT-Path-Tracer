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
    const ReGIRGridFillSurface& cell_surface, Xorshift32Generator& rng)
{
    ReGIRReservoir out_reservoir = current_reservoir;

#if ReGIR_DoVisibilityReuse == KERNEL_OPTION_TRUE
    if (current_reservoir.UCW > 0.0f)
    {
        float3 point_on_light = current_reservoir.sample.point_on_light;

        if (!ReGIR_grid_cell_visibility_test(render_data, cell_surface.cell_point, cell_surface.cell_primitive_index, point_on_light, rng))
            out_reservoir.UCW = ReGIRReservoir::VISIBILITY_REUSE_KILLED_UCW;
    }
#endif

    return out_reservoir;
}

#endif
