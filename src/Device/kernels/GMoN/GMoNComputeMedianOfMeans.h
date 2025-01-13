/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_GMON_H
#define KERNELS_GMON_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#define GMoNThreadsPerBlock (GMoNComputeMeansKernelThreadBlockSize * GMoNComputeMeansKernelThreadBlockSize)

#define ThreadIndex2DTo1D (threadId.x + threadId.y * blockDim.x)
#define SCRATCH_MEMORY_INDEX_DEBUG(input_buffer_index, key_index, threadIndex) (threadIndex + key_index * GMoNThreadsPerBlock + input_buffer_index * GMoNThreadsPerBlock * GMoNMSetsCount)

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

    if (render_data.render_settings.sample_number == 0)
    {
        // For sample 0, this is a special case where we're just going to
        // copy the current pixel color (which is only 1 sample accumulated)
        // to the output framebuffer such that we don't get a black
        // viewport while the full GMoN median of means computation
        // hasn't been launched

        if (pixel_index == 0)
            printf("yes");

        render_data.buffers.gmon_estimator.result_framebuffer[pixel_index] = render_data.buffers.accumulated_ray_colors[pixel_index];
        return;
    }

    if (pixel_index != 0)
        return;

    for (int i = 0; i < 64; i++)
    {
        printf("Tidx %d: %d", i, SCRATCH_MEMORY_INDEX_DEBUG(0, 0, i));

    }

    return;

    ColorRGB32F GMoN_color = render_data.buffers.gmon_estimator.gmon_compute_median_of_means(render_data.buffers.gmon_estimator.sets, pixel_index, render_data.render_settings.sample_number, render_data.render_settings.render_resolution);
    render_data.buffers.gmon_estimator.result_framebuffer[pixel_index] = GMoN_color;
}

#endif
