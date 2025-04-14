/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H
#define HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H

#include "HostDeviceCommon/Color.h"

struct LightSampleInformation
{
    int emissive_triangle_index = -1;
    float3 light_source_normal = { 0.0f, 1.0f, 0.0f };
    float light_area = 1.0f;
    ColorRGB32F emission;

    float3 point_on_light = make_float3(0.0f, 0.0f, 0.0f);
    float area_measure_pdf = 0.0f;
};

#endif
