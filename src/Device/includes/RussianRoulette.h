/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RUSSIAN_ROULETTE_H
#define RUSSIAN_ROULETTE_H

#include "HostDeviceCommon/RenderSettings.h"

/**
 * Returns false if the ray should be killed.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool do_russian_roulette(const HIPRTRenderSettings& render_settings, int bounce, ColorRGB32F& ray_throughput, Xorshift32Generator& random_number_generator)
{
    if (bounce >= render_settings.russian_roulette_min_depth && render_settings.use_russian_roulette)
    {
        float max_throughput = ray_throughput.max_component();
        if (max_throughput >= 1.0f)
            // Never kill these rays because our random in [0, 1]
            // cannot match throughput > 1.0f so this would be biased because
            //
            // - the ray would be never be killed since rand() > throughput is impossible
            // - ray not killed --> throughput /= max_throughput
            // 
            // Which effectively means that we would always be dividing 
            // the throughput of those rays for no reason --> loss of energy
            return true;

        float kill_probability = random_number_generator();
        if (kill_probability > max_throughput)
            // Kill the ray
            return false;

        ray_throughput /= max_throughput;
    }

    // The ray survived
    return true;
}

#endif
