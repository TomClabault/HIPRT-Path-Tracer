/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_SANITY_CHECK_H
#define DEVICE_INCLUDES_SANITY_CHECK_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "Utils/Utils.h" // For debugbreak in sanity_check()

 // For logging stuff on the CPU and avoid everything being mixed
 // up in the terminal because of multithreading
#include <mutex>
std::mutex g_mutex;
#endif

HIPRT_HOST_DEVICE HIPRT_INLINE void debug_set_final_color(const HIPRTRenderData& render_data, int x, int y, int res_x, ColorRGB32F final_color)
{
    if (render_data.render_settings.sample_number == 0)
        render_data.buffers.accumulated_ray_colors[y * res_x + x] = final_color;
    else
        render_data.buffers.accumulated_ray_colors[y * res_x + x] = final_color * render_data.render_settings.sample_number;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_negative_color(ColorRGB32F ray_color, int x, int y, int sample)
{
    (void)x;
    (void)y;
    (void)sample;

    if (ray_color.r < 0 || ray_color.g < 0 || ray_color.b < 0)
    {
#ifndef __KERNELCC__
        std::cout << "Negative color at [" << x << ", " << y << "], sample " << sample << std::endl;
#endif

        return true;
    }

    return false;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_nan(ColorRGB32F ray_color, int x, int y, int sample)
{
    // To avoid unused variables on the GPU
    (void)x;
    (void)y;
    (void)sample;

    if (hippt::is_nan(ray_color.r) || hippt::is_nan(ray_color.g) || hippt::is_nan(ray_color.b))
    {
#ifndef __KERNELCC__
        std::lock_guard<std::mutex> logging_lock(g_mutex);
        std::cout << "NaN at [" << x << ", " << y << "], sample" << sample << std::endl;
#endif
        return true;
    }

    return false;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool sanity_check(const HIPRTRenderData& render_data, RayPayload& ray_payload, int x, int y)
{
    bool invalid = false;
    if (ray_payload.volume_state.sampled_wavelength == 0.0f)
        // Only checking for negative colors if we didn't sample a spectral
        // object because spectral can yield negative values but those are legit
        // and we want to accumulate them
        invalid |= check_for_negative_color(ray_payload.ray_color, x, y, render_data.render_settings.sample_number);
    invalid |= check_for_nan(ray_payload.ray_color, x, y, render_data.render_settings.sample_number);

    if (invalid)
    {
#ifndef __KERNELCC__
        Utils::debugbreak();
#endif

        if (render_data.render_settings.display_NaNs)
            debug_set_final_color(render_data, x, y, render_data.render_settings.render_resolution.x, ColorRGB32F(1.0e30f, 0.0f, 1.0e30f));
        else
            ray_payload.ray_color = ColorRGB32F(0.0f);
    }

    return !invalid;
}

#endif
