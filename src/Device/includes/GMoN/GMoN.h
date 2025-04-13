/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_GMON_H
#define DEVICE_INCLUDES_GMON_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/GMoN/GMoNMeansRadixSort.h"
#include "Device/includes/GMoN/GMoNDevice.h"
#include "HostDeviceCommon/Color.h"

// A bunch of macros here to streamline the code between the CPU and GPU
#ifdef __KERNELCC__

#define SORTED_MEANS_VARIABLE
#define SORTED_MEANS_VARIABLE_WITH_COMMA
// Nothing to declare for the sorted means: on the GPU the sorted means are in shared memory, already declared in 'GMoNMeansRadixSort'
#define SORTED_MEANS_DECLARATION
#define SORTED_MEANS_DECLARATION_WITH_COMMA
#define SORTED_MEANS_ASSIGNATION(x) x
// Getting the sorted mean of index 'mean_index' (in shared memory on the GPU)
#define SORTED_MEANS_FETCH(mean_index) scratch_memory[SCRATCH_MEMORY_INDEX(0, (mean_index))]
#define SORTED_INDEX_FETCH(set_index) (sorted_keys[SORTED_KEYS_INDEX(set_index)] & 0xFF)

#else

// Just a macro for the name of the sorted means std::vector
#define SORTED_MEANS_VARIABLE sorted_means
#define SORTED_MEANS_VARIABLE_WITH_COMMA ,sorted_means
// On the CPU, the sorted means are in a std::vector
#define SORTED_MEANS_DECLARATION std::pair<std::vector<unsigned int>, std::vector<unsigned short int>> SORTED_MEANS_VARIABLE
#define SORTED_MEANS_DECLARATION_WITH_COMMA ,std::pair<std::vector<unsigned int>, std::vector<unsigned short int>> SORTED_MEANS_VARIABLE
// Assigning to the sorted means vector
#define SORTED_MEANS_ASSIGNATION(x) SORTED_MEANS_VARIABLE = (x)
// Getting the sorted mean of index 'mean_index' (in the 'sorted_means' std::vector on the CPU)
#define SORTED_MEANS_FETCH(mean_index) SORTED_MEANS_VARIABLE.first[(mean_index)]
#define SORTED_INDEX_FETCH(set_index) (SORTED_MEANS_VARIABLE.second[set_index] & 0xFF)

#endif

HIPRT_HOST_DEVICE HIPRT_INLINE float compute_gini_coefficient(SORTED_MEANS_DECLARATION)
{
    // Applying Eq. 4 of the paper
    float sum_of_means = 0.0f;
    float sum_of_means_weighted = 0.0f;

    for (int j = 1; j <= GMoNMSetsCount; j++)
    {
        unsigned int sorted_mean_uint = SORTED_MEANS_FETCH(j - 1);
        float sorted_mean_float = *reinterpret_cast<float*>(&sorted_mean_uint);

        sum_of_means += sorted_mean_float;
        sum_of_means_weighted += j * sorted_mean_float;
    }

    float nume = 2.0f * sum_of_means_weighted;
    float denom = GMoNMSetsCount * sum_of_means;

    if (denom == 0.0f)
        return 0.0f;

    return nume / denom - static_cast<float>(GMoNMSetsCount + 1) / GMoNMSetsCount;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F get_median_of_means(GMoNDevice gmon_device, unsigned int pixel_index, int2 render_resolution SORTED_MEANS_DECLARATION_WITH_COMMA)
{
    // Getting the index of the set for the sorted median
    unsigned short int median_set_index = SORTED_INDEX_FETCH(GMoNMSetsCount / 2);

    return gmon_device.sets[median_set_index * render_resolution.x * render_resolution.y + pixel_index];
}

/**
 * Computes the median of means over the sets and stores the
 * result in the 'result_framebuffer' buffer. The result will
 * be stored scaled by the number of samples rendered by the
 * path tracer so far such that dividing the 'result_framebuffer'
 * buffer by the number of samples yields the correct color for
 * displaying in the viewport
 */
HIPRT_HOST_DEVICE ColorRGB32F gmon_compute_median_of_means(GMoNDevice gmon_device, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
    SORTED_MEANS_DECLARATION;
    SORTED_MEANS_ASSIGNATION(gmon_means_radix_sort(gmon_device.sets, pixel_index, sample_number, render_resolution));

    switch (gmon_device.gmon_mode)
    {
    case GMoNDevice::GMoNMode::MEDIAN_OF_MEANS:
        // Multiplying by the number of sets here because (with an example):
        //  - If we have 5 sets
        //  - We rendered 35 samples so far
        //  - Each set has 7 samples
        //  - But the display shader in the viewport expects 35 samples worth of intensity in the framebuffer
        //  - So we need to return the color (which is 7 sample-accumulated) multiplied by the number of sets
        //      to get back our 35
        return get_median_of_means(gmon_device, pixel_index, render_resolution SORTED_MEANS_VARIABLE_WITH_COMMA) * GMoNMSetsCount;

    case GMoNDevice::GMoNMode::BINARY_GMON:
    {
        float gini_coefficient = compute_gini_coefficient(SORTED_MEANS_VARIABLE);

        // Eq. 5 of the paper
        if (gini_coefficient <= 0.25f)
        {
            // Return the mean. We're actually just going to return the sum of the samples and it is the shader that
            // displays in the viewport that is going to divide that sum by the number of samples rendered so far,
            // thus giving us the mean
            
            ColorRGB32F sum;
            for (int i = 0; i < GMoNMSetsCount; i++)
                sum += gmon_device.sets[render_resolution.x * render_resolution.y * i + pixel_index];

            return sum;
        }
        else
            // Return the median of means
             
            // Multiplying by the number of sets here because (with an example):
            //  - If we have 5 sets
            //  - We rendered 35 samples so far
            //  - Each set has 7 samples
            //  - But the display shader in the viewport expects 35 samples worth of intensity in the framebuffer
            //  - So we need to return the color (which is 7 sample-accumulated) multiplied by the number of sets
            //      to get back our 35
            return get_median_of_means(gmon_device, pixel_index, render_resolution SORTED_MEANS_VARIABLE_WITH_COMMA) * GMoNMSetsCount;
    }

    case GMoNDevice::GMoNMode::ADAPTIVE_GMON:
    {
        // Section 4.3 and Eq. 6
        float gini_coefficient = compute_gini_coefficient(SORTED_MEANS_VARIABLE);
        if (gini_coefficient == 0.0f)
            return ColorRGB32F(0.0f);

        int c = gini_coefficient * (GMoNMSetsCount / 2);

        // Eq. 6
        ColorRGB32F sum;
        for (int i = c; i < GMoNMSetsCount - c; i++)
            sum += gmon_device.sets[SORTED_INDEX_FETCH(i) * render_resolution.x * render_resolution.y + pixel_index];

        // We want this function to return un-averaged colors such that it is
        // the shader that displays in the viewport that does the averaging.
        // That's why we have the multiplication by GMoNMSetsCount at the end (which isn't in the paper)
        return sum / (GMoNMSetsCount - 2 * c) * GMoNMSetsCount;
    }

    default:
        // We shouldn't be here, this means that the GMoNMode used isn't one of the GMoNMode enum
        return ColorRGB32F(10000.0f, 0.0f, 10000.0f);
    }
 
    // We cannot be here, this would mean that the switch skipped every single case, including the default case
    return ColorRGB32F(10000.0f, 0.0f, 0.0f);
}

#endif
