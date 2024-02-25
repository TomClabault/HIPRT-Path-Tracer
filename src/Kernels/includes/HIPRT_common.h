#ifndef HIPRTRT_COMMON
#define HIPRTRT_COMMON

#include "HostDeviceCommon/color.h"
#include "HostDeviceCommon/material.h"
#include "Kernels/includes/hiprt_fix_vs.h"
#include "Kernels/includes/HIPRT_maths.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

struct HIPRT_xorshift32_state {
    unsigned int a = 42;
};

struct HIPRT_xorshift32_generator
{
    //__device__ HIPRT_xorshift32_generator(unsigned int seed) : m_state({ seed }) {}

    __device__ float operator()()
    {
        //Float in [0, 1[
        return RT_MIN(xorshift32() / (float)UINT_MAX, 1.0f - 1.0e-6f);
    }

    __device__ unsigned int xorshift32()
    {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        unsigned int x = m_state.a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return m_state.a = x;
    }

    HIPRT_xorshift32_state m_state;
};

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
