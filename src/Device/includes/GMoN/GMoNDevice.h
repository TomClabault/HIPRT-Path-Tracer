/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GMON_DEVICE_H
#define DEVICE_GMON_DEVICE_H

#include "Device/includes/GMoN/GMoNMeansRadixSort.h"
#include "HostDeviceCommon/Color.h"

/**
 * Data structure for the implementation of GMoN
 * 
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */
struct GMoNDevice
{
    /**
     * Computes the median of means over the sets and stores the 
     * result in the 'result_framebuffer' buffer. The result will 
     * be stored scaled by the number of samples rendered by the 
     * path tracer so far such that dividing the 'result_framebuffer' 
     * buffer by the number of samples yields the correct color for 
     * displaying in the viewport
     */
    HIPRT_HOST_DEVICE void compute_gmon(ColorRGB32F* gmon_sets, ColorRGB32F* out_gmon_framebuffer, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution) const
    {
        ColorRGB32F gmon_median_of_means = gmon_compute_median_of_means(gmon_sets, pixel_index, sample_number, render_resolution);

        out_gmon_framebuffer[pixel_index] = gmon_median_of_means;
    }

    // This is one very big buffer that contains all the sets we accumulate into for GMoN
    //
    // For example, for GMoNCPUGPUCommonData::number_of_sets == 5 and a render resoltuion of 1280x720,
    // this is going to be a buffer that is 1280*720*5 elements long
    ColorRGB32F* sets = nullptr;

    // This is the buffer that contains the G-median of means result of each pixel and this is going
    // to be displayed in the viewport instead of the regular framebuffer if GMoN is being used
    ColorRGB32F* result_framebuffer = nullptr;

    // Which is the next set that is going to receive the sample
    unsigned int next_set_to_accumulate = 0;

    unsigned int number_of_sets = 5;

private:
    HIPRT_HOST_DEVICE ColorRGB32F gmon_compute_median_of_means(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution) const
    {
#ifdef __KERNELCC__
        unsigned int median = scratch_memory[SCRATCH_MEMORY_INDEX(0, GMON_M_SETS_COUNT / 2)];
#else
        std::vector<unsigned int> sorted_means = gmon_means_radix_sort(gmon_sets, pixel_index, sample_number, render_resolution);
        unsigned int median = sorted_means[GMON_M_SETS_COUNT / 2];
#endif

        // The median is in the middle of the vector
        float median_float = *reinterpret_cast<float*>(&median);
        unsigned int sample_scaling = sample_number / (GMON_M_SETS_COUNT);

        // Now finding what color had that median
        for (int i = 0; i < GMON_M_SETS_COUNT; i++)
        {
            ColorRGB32F color = gmon_sets[render_resolution.x * render_resolution.y * i + pixel_index];
            if (color.luminance() / sample_scaling == median_float)
                return color;
        }

        // We should never be here, this would mean that the median found in the means sets wasnt in the sets in the first place
        return ColorRGB32F(0.0f);
    }
};

#endif
