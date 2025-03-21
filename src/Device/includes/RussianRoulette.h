/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RUSSIAN_ROULETTE_H
#define RUSSIAN_ROULETTE_H

#include "HostDeviceCommon/RenderSettings.h"

/**
 * Returns false if the ray should be killed.
 * 
 * This overload returns in 'throughput_scaling' the throughput scaling that
 * has been applied to 'ray_throughput' to account for the russian roulette surviving
 * probability
 * 
 * If russian roulette wasn't applied, 'throughput_scaling' is left untouched
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool do_russian_roulette(const HIPRTRenderSettings& render_settings, int bounce, ColorRGB32F& ray_throughput, float& throughput_scaling, const ColorRGB32F& current_weight, Xorshift32Generator& random_number_generator)
{
    if (bounce >= render_settings.russian_roulette_min_depth && render_settings.do_russian_roulette)
    {
        float survive_probability = 0.0f;
        if (render_settings.path_russian_roulette_method == PathRussianRoulette::MAX_THROUGHPUT)
            // Easy max throughput threshold
            survive_probability = ray_throughput.max_component();
        else if (render_settings.path_russian_roulette_method == PathRussianRoulette::ARNOLD_2014)
        {
            // Reference:
            // [Physically Based Shader Design in Arnold, Langlands, 2014]
            survive_probability = (ray_throughput * current_weight).max_component() / ray_throughput.max_component();
            survive_probability = sqrtf(survive_probability);
        }

        // Clamping anything above one back to 1
        survive_probability = hippt::min(survive_probability, 1.0f);

        if (random_number_generator() > survive_probability)
            // Kill the ray
            return false;

        throughput_scaling = 1.0f / survive_probability;
        if (render_settings.russian_roulette_throughput_clamp > 0.0f)
            // Clamping the throughput increase to avoid fireflies by
            // rays that still pass the russian roulette with very low
            // probabilities
            throughput_scaling = hippt::min(throughput_scaling, render_settings.russian_roulette_throughput_clamp);

        ray_throughput *= throughput_scaling;
    }

    // The ray survived
    return true;
}

/**
 * Returns false if the ray should be killed.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool do_russian_roulette(const HIPRTRenderSettings& render_settings, int bounce, ColorRGB32F& ray_throughput, const ColorRGB32F& current_weight, Xorshift32Generator& random_number_generator)
{
    float unused_throughput_scaling;
    return do_russian_roulette(render_settings, bounce, ray_throughput, unused_throughput_scaling, current_weight, random_number_generator);
}

#endif
