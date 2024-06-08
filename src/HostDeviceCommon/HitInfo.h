/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RAY_STATE_H
#define RAY_STATE_H

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
    float3 inter_point;
    float3 shading_normal;
    float3 geometric_normal;
    float2 texcoords;
    float2 uv;

    float t = -1.0f; // Distance along ray

    int primitive_index = -1;
};

#endif