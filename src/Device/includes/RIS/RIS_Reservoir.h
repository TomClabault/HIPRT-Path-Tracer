/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RIS_RESERVOIR_H
#define DEVICE_RIS_RESERVOIR_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Xorshift.h"

#ifndef __KERNELCC__
#include "Utils/Utils.h"

 // For multithreaded console error logging on the CPU if NaNs are detected
#include <mutex>
static std::mutex ris_log_mutex;
#endif

struct RISSample
{
    int emissive_triangle_index = -1;
    float3 point_on_light_source = { 0, 0, 0 };

    float target_function = 0.0f;

    // TODO Can this be refactored? Is this needed?
    bool is_bsdf_sample = false;
    ColorRGB32F bsdf_sample_contribution;
    float bsdf_sample_cosine_term = 0.0f;
};

struct RISReservoir
{
    HIPRT_HOST_DEVICE void add_one_candidate(RISSample new_sample, float weight, Xorshift32Generator& random_number_generator)
    {
        M++;
        weight_sum += weight;

        if (random_number_generator() < weight / weight_sum)
            sample = new_sample;
    }

    HIPRT_HOST_DEVICE void end()
    {
        if (weight_sum == 0.0f)
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum;
    }

    HIPRT_HOST_DEVICE HIPRT_INLINE void sanity_check(int2 pixel_coords = make_int2(-1, -1))
    {
#ifndef __KERNELCC__
        if (M < 0)
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "Negative reservoir M value at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << M << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(weight_sum) || std::isinf(weight_sum))
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "NaN or inf reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (weight_sum < 0)
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "Negative reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
            Utils::debugbreak();
        }
        else if (std::abs(weight_sum) < std::numeric_limits<float>::min() && weight_sum != 0.0f)
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "Denormalized weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(UCW) || std::isinf(UCW))
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "NaN or inf reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (UCW < 0)
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "Negative reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << UCW << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(sample.target_function) || std::isinf(sample.target_function))
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "NaN or inf reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (sample.target_function < 0)
        {
            std::lock_guard<std::mutex> lock(ris_log_mutex);
            std::cerr << "Negative reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << sample.target_function << std::endl;
            Utils::debugbreak();
        }
#else
        (void)pixel_coords;
#endif
    }

    unsigned int M = 0;
    // TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
    float weight_sum = 0.0f;
    float UCW = 0.0f;

    RISSample sample;
};

#endif
