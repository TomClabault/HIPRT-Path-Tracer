/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_GMON_H
#define KERNELS_GMON_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) GMoNComputeMedianOfMeans(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline GMoNComputeMedianOfMeans(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    render_data.buffers.gmon_estimator.compute_gmon(render_data.buffers.gmon_estimator.sets, render_data.buffers.gmon_estimator.result_framebuffer, pixel_index, render_data.render_settings.sample_number, render_data.render_settings.render_resolution);
}

#endif