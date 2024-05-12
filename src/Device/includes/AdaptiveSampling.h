/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef ADAPTIVE_SAMPLING_H
#define ADAPTIVE_SAMPLING_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE bool adaptive_sampling(const HIPRTRenderData& render_data, int pixel_index)
{
    int pixel_sample_count = render_data.aux_buffers.pixel_sample_count[pixel_index];
    if (pixel_sample_count < 0)
        // Pixel is deactivated
        return false;

    if (pixel_sample_count > render_data.render_settings.adaptive_sampling_min_samples)
    {
        // Waiting for at least 16 samples to enable adaptive sampling
        float luminance = render_data.buffers.pixels[pixel_index].luminance();

        float average_luminance = luminance / (pixel_sample_count + 1);
        float squared_luminance = render_data.aux_buffers.pixel_squared_luminance[pixel_index];

        float pixel_variance = (squared_luminance - luminance * average_luminance) / (pixel_sample_count);

        bool pixel_needs_sampling = 1.96f * sqrtf(pixel_variance) / sqrtf(pixel_sample_count + 1) > render_data.render_settings.adaptive_sampling_noise_threshold * average_luminance;
        if (!pixel_needs_sampling)
        {
            // Indicates no need to sample anymore by setting the sample count to negative
            render_data.aux_buffers.pixel_sample_count[pixel_index] = -pixel_sample_count;

            return false;
        }
    }

    return true;
}

#endif
