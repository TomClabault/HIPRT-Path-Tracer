/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

HIPRT_HOST_DEVICE HIPRT_INLINE void debug_set_final_color(const HIPRTRenderData& render_data, int x, int y, ColorRGB32F final_color)
{
    if (render_data.render_settings.sample_number == 0)
        render_data.buffers.accumulated_ray_colors[y * render_data.render_settings.render_resolution.x + x] = final_color;
    else
        render_data.buffers.accumulated_ray_colors[y * render_data.render_settings.render_resolution.x + x] = final_color * render_data.render_settings.sample_number;
}

/**
 * Returns true if the color has a negative component.
 * False otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_negative_color(ColorRGB32F ray_color, int x, int y, int sample)
{
    // To remove 'unused variable' warnings of the GPU compiler because these variables are only used
    // in the std::cout of the CPU
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

/**
 * Returns true if the color has a NaN or INF component.
 * False otherwise
 */ 
HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_nan(ColorRGB32F ray_color, int x, int y, int sample)
{
    // To avoid unused variables on the GPU
    (void)x;
    (void)y;
    (void)sample;

    if (hippt::is_nan(ray_color.r) || hippt::is_nan(ray_color.g) || hippt::is_nan(ray_color.b) ||
        hippt::is_inf(ray_color.r) || hippt::is_inf(ray_color.g) || hippt::is_inf(ray_color.b))
    {
#ifndef __KERNELCC__
        std::lock_guard<std::mutex> logging_lock(g_mutex);
        std::cout << "NaN/INF at [" << x << ", " << y << "], sample" << sample << std::endl;
#endif
        return true;
    }

    return false;
}

template <bool CheckOnlyOnCPU = false>
HIPRT_HOST_DEVICE HIPRT_INLINE bool sanity_check(const HIPRTRenderData& render_data, ColorRGB32F& in_out_color, int x, int y)
{
    if constexpr (CheckOnlyOnCPU)
    {
#ifdef __KERNELCC__
        return true;
#endif
    }

    bool valid = true;

    valid &= !check_for_negative_color(in_out_color, x, y, render_data.render_settings.sample_number);
    valid &= !check_for_nan(in_out_color, x, y, render_data.render_settings.sample_number);

    if (!valid)
    {
#ifndef __KERNELCC__
        Utils::debugbreak();
#endif

        if (render_data.render_settings.display_NaNs)
            debug_set_final_color(render_data, x, y, ColorRGB32F(1.0e30f, 0.0f, 1.0e30f));
        else
            in_out_color = ColorRGB32F(0.0f);
    }

    return valid;
}

template <bool CheckOnlyOnCPU = false>
HIPRT_HOST_DEVICE HIPRT_INLINE bool sanity_check(const HIPRTRenderData& render_data, const ColorRGB32F& in_out_color, int x, int y)
{
    ColorRGB32F copy = in_out_color;
    return sanity_check<CheckOnlyOnCPU>(render_data, copy, x, y);
}

#endif
