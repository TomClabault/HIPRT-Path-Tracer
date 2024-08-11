/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ADAPTIVE_SAMPLING_H
#define DEVICE_ADAPTIVE_SAMPLING_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float get_pixel_confidence_interval(const HIPRTRenderData& render_data, int pixel_index, int pixel_sample_count, float& average_luminance)
{
    float luminance = render_data.buffers.pixels[pixel_index].luminance();
    average_luminance = luminance / (pixel_sample_count + 1);

    float squared_luminance = render_data.aux_buffers.pixel_squared_luminance[pixel_index];
    float pixel_variance = (squared_luminance - luminance * average_luminance) / (pixel_sample_count);

    return 1.96f * sqrtf(pixel_variance) / sqrtf(pixel_sample_count + 1);
}

/**
 * pixel_converged is set to true if the given pixel has reached the noise
 * threshold given in render_data.render_settings.stop_pixel_percentage_converged. It 
 * is set to false otherwise.
 * 
 * Returns true if the pixel needs more sample according to adaptive sampling (or if adaptive sampling is disabled).
 * Returns false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool adaptive_sampling(const HIPRTRenderData& render_data, int pixel_index, bool& pixel_converged)
{
    const HIPRTRenderSettings& render_settings = render_data.render_settings;
    const AuxiliaryBuffers& aux_buffers = render_data.aux_buffers;

    if (render_settings.stop_pixel_noise_threshold > 0.0f)
    {
        // The stopping condition on the proportion of
        // pixels having converged is enabled (stop_pixel_percentage_converged)
        // 
        // We're going to compute the erorr of the pixel to see if it's below the threshold or not

        int pixel_sample_count;
        float average_luminance;
        float confidence_interval;
        pixel_sample_count = aux_buffers.pixel_sample_count[pixel_index];
        confidence_interval = get_pixel_confidence_interval(render_data, pixel_index, pixel_sample_count, average_luminance);

        // The value of pixel_converged will be used outside of this function
        pixel_converged =
            // Converged enough
            (confidence_interval < render_settings.stop_pixel_noise_threshold * average_luminance)
            // At least 2 samples because the maths break down at 1 sample
            && (render_settings.sample_number > 1);
    }

    if (render_settings.enable_adaptive_sampling)
    {
        // Computing pixel convergence according to adaptive sampling to
        // know whether to keep sampling that pixel or not

        int pixel_sample_count = aux_buffers.pixel_sample_count[pixel_index];
        if (pixel_sample_count < 0)
            // Pixel is deactivated
            return false;

        if (pixel_sample_count > render_settings.adaptive_sampling_min_samples)
        {
            bool pixel_needs_sampling;
            float average_luminance;
            float confidence_interval;
            confidence_interval = get_pixel_confidence_interval(render_data, pixel_index, pixel_sample_count, average_luminance);

            pixel_needs_sampling = confidence_interval > render_settings.adaptive_sampling_noise_threshold * average_luminance;
            if (!pixel_needs_sampling)
            {
                // Indicates no need to sample anymore by setting the sample count to negative
                aux_buffers.pixel_sample_count[pixel_index] = -pixel_sample_count;

                return false;
            }
        }

        return true;
    }

    return true;
}

#endif
