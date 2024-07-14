/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIT_INFO_H
#define HIT_INFO_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"

struct LightSourceInformation
{
    int emissive_triangle_index = -1;
    float3 light_source_normal = { 0.0f, 1.0f, 0.0f };
    float light_area = 1.0f;
    ColorRGB emission;
};

struct HitInfo
{
    float3 inter_point = { 0, 0, 0 };
    float3 shading_normal = { 0, 0, 0 };
    float3 geometric_normal = { 0, 0, 0 };
    float2 texcoords = { 0, 0 };
    float2 uv = { 0, 0 };

    // Distance along ray
    float t = -1.0f;

    int primitive_index = -1;
};

#endif