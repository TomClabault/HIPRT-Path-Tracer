/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESERVOIR_H
#define DEVICE_RESERVOIR_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Xorshift.h"

struct ReservoirSample
{
    float3 point_on_light_source = { 0, 0, 0 };
    float3 light_source_normal = { 0, 0, 0 };
    ColorRGB emission = { 0.0f, 0.0f, 0.0f };

    float target_function = 0.0f;
};

struct Reservoir
{
    HIPRT_HOST_DEVICE void add_one_candidate(ReservoirSample new_sample, float weight, Xorshift32Generator& random_number_generator)
    {
        M++;
        weight_sum += weight;

        if (random_number_generator() < weight / weight_sum)
            sample = new_sample;
    }

    HIPRT_HOST_DEVICE void combine_with(Reservoir other_reservoir, float r_m, float target_function, Xorshift32Generator& random_number_generator)
    {
        // ReSTIR 2019, Alg. 6, line 4
        // pHat_q(r.y) * r.W * r.M
        // target_function = 
        float reservoir_sample_weight = target_function * other_reservoir.UCW * r_m;

        M += other_reservoir.M;
        weight_sum += reservoir_sample_weight;

        if (random_number_generator() < reservoir_sample_weight / weight_sum)
        {
            sample = other_reservoir.sample;
            sample.target_function = target_function;

            debug_value = other_reservoir.UCW;
        }
    }

    HIPRT_HOST_DEVICE void end()
    {
        if (weight_sum == 0.0f)
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum;
    }

    HIPRT_HOST_DEVICE void end_normalized(float Z)
    {
        if (weight_sum == 0.0f || Z == 0.0f)
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum / Z;
    }

    unsigned int M = 0;
    // TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
    float weight_sum = 0.0f;
    float UCW = 0.0f;

    // This debug value stored in the reservoir can be used to display
    // a value on the viewport such as the UCW for example or something else
    float debug_value = 0.0f;

    ReservoirSample sample;
};

#endif
