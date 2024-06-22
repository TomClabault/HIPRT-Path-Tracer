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
    ColorRGB emission = { 0.0f, 0.0f, 0.0f };

    float target_function = 0.0f;
};

struct Reservoir
{
    HIPRT_HOST_DEVICE void add_one_candidate(ReservoirSample new_sample, float weight, Xorshift32Generator& random_number_generator)
    {
        M++;
        weight_sum += weight;

        if (random_number_generator() < (weight / weight_sum))
            sample = new_sample;
    }

    HIPRT_HOST_DEVICE void combine_with(Reservoir other_reservoir, float mis_weight, float target_function, Xorshift32Generator& random_number_generator)
    {
        float reservoir_sample_weight = mis_weight * target_function * other_reservoir.UCW * other_reservoir.M;

        M += other_reservoir.M;
        weight_sum += reservoir_sample_weight;

        if (random_number_generator() < (reservoir_sample_weight / weight_sum))
        {
            sample = other_reservoir.sample;
            sample.target_function = target_function;
        }
    }

    HIPRT_HOST_DEVICE void end()
    {
        if (weight_sum == 0.0f)
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum;
    }

    unsigned int M = 0;
    float weight_sum = 0.0f;
    float UCW = 0.0f;

    ReservoirSample sample;
};

#endif
