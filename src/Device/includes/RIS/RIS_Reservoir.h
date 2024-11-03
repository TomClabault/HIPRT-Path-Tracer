/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RIS_RESERVOIR_H
#define DEVICE_RIS_RESERVOIR_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Xorshift.h"

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
        if (hippt::isZERO(weight_sum))
            UCW = 0.0f;
        else
            UCW = 1.0f / sample.target_function * weight_sum;
    }

    unsigned int M = 0;
    // TODO weight sum is never used at the same time as UCW so only one variable can be used for both to save space
    float weight_sum = 0.0f;
    float UCW = 0.0f;

    RISSample sample;
};

#endif
