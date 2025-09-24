/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_HIT_INFO_H
#define HOST_DEVICE_COMMON_HIT_INFO_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"

struct HitInfo
{
    float3 inter_point = { 0, 0, 0 };
    float3 shading_normal = { 0, 0, 0 };
    float3 geometric_normal = { 0, 0, 0 };
    // TODO is texcoords useful? This may actually be returned by the intersection function and used only for reading textures but then we don't need it anymore when evaluating the bSDF and comp�ting the main path tracing stuff so let's save some registers
    float2 texcoords = { 0, 0 };

    // Distance along ray
    float t = -1.0f;

    int primitive_index = -1;
};

#endif
