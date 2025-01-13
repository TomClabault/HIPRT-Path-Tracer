/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_GMON_H
#define DEVICE_INCLUDES_GMON_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/GMoN/GMoNMeansRadixSort.h"
#include "HostDeviceCommon/Color.h"

/**
 * Computes the median of means over the sets and stores the
 * result in the 'result_framebuffer' buffer. The result will
 * be stored scaled by the number of samples rendered by the
 * path tracer so far such that dividing the 'result_framebuffer'
 * buffer by the number of samples yields the correct color for
 * displaying in the viewport
 */
HIPRT_HOST_DEVICE ColorRGB32F gmon_compute_median_of_means(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
#ifdef __KERNELCC__
    gmon_means_radix_sort(gmon_sets, pixel_index, sample_number, render_resolution);
    unsigned int median = scratch_memory[SCRATCH_MEMORY_INDEX(0, GMoNMSetsCount / 2)];
#else
    std::vector<unsigned int> sorted_means = gmon_means_radix_sort(gmon_sets, pixel_index, sample_number, render_resolution);
    unsigned int median = sorted_means[GMoNMSetsCount / 2];
#endif

    // The median is in the middle of the vector
    float median_float = *reinterpret_cast<float*>(&median);
    unsigned int sample_scaling = sample_number / (GMoNMSetsCount);

    // Now finding what color had that median
    for (int i = 0; i < GMoNMSetsCount; i++)
    {
        ColorRGB32F color = gmon_sets[render_resolution.x * render_resolution.y * i + pixel_index];
        if (color.luminance() / sample_scaling == median_float)
            return color * GMoNMSetsCount;
    }

    // We should never be here, this would mean that the median found in the means sets wasnt in the sets in the first place
    return ColorRGB32F(10000.0f, 0.0f, 10000.0f);
}

#endif
