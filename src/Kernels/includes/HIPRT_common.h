#ifndef HIPRTRT_COMMON
#define HIPRTRT_COMMON

#include "HostDeviceCommon/color.h"
#include "HostDeviceCommon/material.h"
#include "HostDeviceCommon/xorshift.h"
#include "Kernels/includes/hiprt_fix_vs.h"
#include "Kernels/includes/HIPRT_maths.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

struct HIPRTLightSourceInformation
{
    int emissive_triangle_index = -1;
    hiprtFloat3 light_source_normal;
};

enum HIPRTRayState
{
    HIPRT_BOUNCE,
    HIPRT_MISSED,
    HIPRT_TERMINATED
};

struct HIPRTHitInfo
{
    hiprtFloat3 inter_point;
    hiprtFloat3 normal_at_intersection;
    hiprtFloat2 uv;

    float t = -1.0f; //Distance along ray

    int primitive_index = -1;
};

#endif // !HIPRTRT_COMMON
