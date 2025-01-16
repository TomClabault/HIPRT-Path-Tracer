/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_GMON_H
#define KERNELS_GMON_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/GMoN/GMoN.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Kernel for the implementation of GMoN
 *
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */

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

    // - 1 here because 'minimum_number_of_samples' is controlled by the user and it starts at 1
    // whereas 'render_data.render_settings.sample_number' starts at 0
    //if (render_data.render_settings.sample_number <= render_data.buffers.gmon_estimator.minimum_number_of_samples - 1)
    if (render_data.render_settings.sample_number == 0)
    {
        // For sample 0, this is a special case where we're just going to
        // copy the current pixel color (which is only 1 sample accumulated)
        // to the output framebuffer such that we don't get a black
        // viewport while the full GMoN median of means computation
        // hasn't been launched
        render_data.buffers.gmon_estimator.result_framebuffer[pixel_index] = render_data.buffers.accumulated_ray_colors[pixel_index];

        return;
    }

    ColorRGB32F GMoN_color = gmon_compute_median_of_means(render_data.buffers.gmon_estimator, pixel_index, render_data.render_settings.sample_number, render_data.render_settings.render_resolution);
    ColorRGB32F non_GMoN_color = render_data.buffers.accumulated_ray_colors[pixel_index];

    ColorRGB32F blended_color = hippt::lerp(non_GMoN_color, GMoN_color, render_data.buffers.gmon_estimator.gmon_blend);
    render_data.buffers.gmon_estimator.result_framebuffer[pixel_index] = GMoN_color;
}

#endif
