/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_RESERVOIR_H
#define DEVICE_RESTIR_DI_RESERVOIR_H

#include "Device/includes/ReSTIR/DI/SampleFlags.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Xorshift.h"

#ifndef __KERNELCC__
#include "Utils/Utils.h"

// For multithreaded console error logging on the CPU if NaNs are detected
#include <mutex>
static std::mutex log_mutex;
#endif

struct ReSTIRDISample
{
    // Global primitive index corresponding to the emissive triangle sampled
    int emissive_triangle_index = -1;

    // For envmap samples, this 'point_on_light_source' is the envmap direction in *envmap space*
    // A sample is an envmap sample if 'flags' contains 'RESTIR_DI_FLAGS_ENVMAP_SAMPLE'
    float3 point_on_light_source = { 0, 0, 0 };

    float target_function = 0.0f;

    // Some flags about the sample
    unsigned char flags = RESTIR_DI_FLAGS_NONE;
};

struct ReSTIRDIReservoir
{
    HIPRT_HOST_DEVICE void add_one_candidate(ReSTIRDISample new_sample, float weight, Xorshift32Generator& random_number_generator)
    {
        M++;
        weight_sum += weight;

        if (random_number_generator() < weight / weight_sum)
            sample = new_sample;
    }

    /**
     * Combines 'other_reservoir' into this reservoir
     * 
     * 'target_function' is the target function evaluated at the pixel that is doing the
     *      resampling with the sample from the reservoir that we're combining (which is 'other_reservoir')
     * 
     * 'jacobian_determinant' is the determinant of the jacobian. In ReSTIR DI, it is used
     *      for converting the solid angle PDF (or UCW since the UCW is an estimate of the PDF)
     *      with respect to the shading point of the reservoir we're resampling to the solid
     *      angle PDF with respect to the shading point of 'this' reservoir
     * 
     * 'random_number_generator' for generating the random number that will be used to stochastically
     *      select the sample from 'other_reservoir' or not
     */
    HIPRT_HOST_DEVICE bool combine_with(ReSTIRDIReservoir other_reservoir, float mis_weight, float target_function, float jacobian_determinant, Xorshift32Generator& random_number_generator)
    {
        if (other_reservoir.UCW <= 0.0f)
        {
            // Not going to be resampled anyways because of invalid UCW so quit exit
            M += other_reservoir.M;

            return false;
        }

        float reservoir_sample_weight = mis_weight * target_function * other_reservoir.UCW * jacobian_determinant;

        M += other_reservoir.M;
        weight_sum += reservoir_sample_weight;

        if (random_number_generator() < reservoir_sample_weight / weight_sum)
        {
            sample = other_reservoir.sample;
            sample.target_function = target_function;

            return true;
        }

        return false;
    }

    HIPRT_HOST_DEVICE void end()
    {
        if (weight_sum == 0.0f)
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum;
    }

    HIPRT_HOST_DEVICE void end_with_normalization(float normalization_numerator, float normalization_denominator)
    {
        // Checking some limit values
        if (weight_sum == 0.0f || weight_sum < 1.0e-10f || weight_sum > 1.0e10f || normalization_denominator == 0.0f || normalization_numerator == 0.0f)
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum * normalization_numerator / normalization_denominator;

        // Hard limiting M to avoid explosions if the user decides not to use any M-cap (M-cap == 0)
        M = hippt::min(M, 1000000);
    }

    HIPRT_HOST_DEVICE HIPRT_INLINE void sanity_check(int2 pixel_coords)
    {
#ifndef __KERNELCC__
        if (M < 0)
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "Negative reservoir M value at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << M << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(weight_sum) || std::isinf(weight_sum))
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "NaN or inf reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (weight_sum < 0)
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "Negative reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
            Utils::debugbreak();
        }
        else if (std::abs(weight_sum) < std::numeric_limits<float>::min() && weight_sum != 0.0f)
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "Denormalized weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(UCW) || std::isinf(UCW))
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "NaN or inf reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (UCW < 0)
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "Negative reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << UCW << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(sample.target_function) || std::isinf(sample.target_function))
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "NaN or inf reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (sample.target_function < 0)
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "Negative reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << sample.target_function << std::endl;
            Utils::debugbreak();
        }
#else
        (void)pixel_coords;
#endif
    }

    int M = 0;
    // TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
    float weight_sum = 0.0f;
    float UCW = 0.0f;

    ReSTIRDISample sample;
};

#endif
