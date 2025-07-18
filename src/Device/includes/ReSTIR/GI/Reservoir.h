/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_RESERVOIR_H
#define DEVICE_RESTIR_GI_RESERVOIR_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Xorshift.h"

#ifndef __KERNELCC__
#include "Utils/Utils.h"

// For multithreaded console error logging on the CPU if NaNs are detected
#include <mutex>
static std::mutex restir_gi_log_mutex;
#endif

struct ReSTIRGISample
{
    float3 sample_point = make_float3(-1.0f, -1.0f, -1.0f);

    int sample_point_primitive_index = -1;

    RGBE9995Packed incoming_radiance_to_visible_point;

    BSDFIncidentLightInfo incident_light_info_at_visible_point = BSDFIncidentLightInfo::NO_INFO;

    // TODO is this one needed? I guess we're going to get a bunch of wrong shading where a sample was resampled and at shading time it hits an alpha geometry where that alpha geometry let the ray through at initial candidates sampling time. This should be unbiased? Maybe not actually. But is it that bad?
    unsigned int visible_to_sample_point_alpha_test_random_seed = 42;

    // TODO can be stored in outoging_radiance_to_first_hit?
    float target_function = 0.0f;

    // Whether or not the sample point is on a material that is rough enough to be reconnected
    // If the sample point is on a mirror for example, reconnecting to that point from our center pixel
    // is going to change the view direction of the mirror BSDF without changing the incident light
    // direction of the mirror BSDF and that's not going to work
    //
    // Also, because we do not re-evaluate the BSDF at the sample point, this would lead to some brightening
    // bias because this would be assuming that reconnecting to the mirror has non-zero energy, even with
    // the new view direction which is incorrect
    bool sample_point_rough_enough = false;

    Octahedral24BitNormalPadded32b sample_point_geometric_normal;

    HIPRT_HOST_DEVICE bool is_envmap_path() const {  return sample_point_primitive_index == -1; }
};

struct ReSTIRGIReservoir
{
    HIPRT_HOST_DEVICE void add_one_candidate(ReSTIRGISample new_sample, float weight, Xorshift32Generator& random_number_generator)
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
    HIPRT_HOST_DEVICE bool combine_with(const ReSTIRGIReservoir& other_reservoir, float mis_weight, float target_function, float jacobian_determinant, Xorshift32Generator& random_number_generator)
    {
        // Bullet point 4. of the intro of Section 5.2 of [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
        float reservoir_resampling_weight = mis_weight * target_function * other_reservoir.UCW * jacobian_determinant;

        weight_sum += reservoir_resampling_weight;
        M += other_reservoir.M;

        if (random_number_generator() < reservoir_resampling_weight / weight_sum)
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
        if (weight_sum == 0.0f || weight_sum > 1.0e10f || normalization_denominator == 0.0f || normalization_numerator == 0.0f)
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
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "Negative reservoir M value at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << M << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(weight_sum) || std::isinf(weight_sum))
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "NaN or inf reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (weight_sum < 0)
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "Negative reservoir weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
            Utils::debugbreak();
        }
        else if (std::abs(weight_sum) < std::numeric_limits<float>::min() && weight_sum != 0.0f)
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "Denormalized weight_sum at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << weight_sum << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(UCW) || std::isinf(UCW))
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "NaN or inf reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (UCW < 0)
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "Negative reservoir UCW at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << UCW << std::endl;
            Utils::debugbreak();
        }
        else if (std::isnan(sample.target_function) || std::isinf(sample.target_function))
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "NaN or inf reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << ")" << std::endl;
            Utils::debugbreak();
        }
        else if (sample.target_function < 0)
        {
            std::lock_guard<std::mutex> lock(restir_gi_log_mutex);
            std::cerr << "Negative reservoir sample.target_function at pixel (" << pixel_coords.x << ", " << pixel_coords.y << "): " << sample.target_function << std::endl;
            Utils::debugbreak();
        }
#else
        (void)pixel_coords;
#endif
    }

    ReSTIRGISample sample;

    int M = 0;
    // TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
    float weight_sum = 0.0f;
    // If the UCW is set to -1, this is because the reservoir was killed by visibility reuse
    float UCW = 0.0f;
};

#endif
