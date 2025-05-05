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
        //if (!ReGIR_non_shading_evaluate_target_function<true>(render_data, hash_grid_cell_index, current_reservoir.sample.emission, current_reservoir.sample.point_on_light, rng))
        if (!ReGIR_grid_cell_visibility_test(render_data, hash_grid_cell_index, current_reservoir.sample.point_on_light, rng))
            out_reservoir.UCW = ReGIRReservoir::VISIBILITY_REUSE_KILLED_UCW;
#endif

    return out_reservoir;
}

#endif
