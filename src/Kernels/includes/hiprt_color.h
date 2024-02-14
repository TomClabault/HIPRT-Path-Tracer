
#ifndef HIPRT_COLOR_H
#define HIPRT_COLOR_H

#include <hiprt/hiprt_device.h>

#include "Kernels/includes/hiprt_fix_vs.h"
#include "Kernels/includes/HIPRT_maths.h"

// TODO instead of duplicating the structure, it would be better to create a folder
// HostDeviceCommon containing the structures that are used both by the GPU and CPU renderer
struct HIPRTColor
{
    __device__ HIPRTColor(float value) : r(value), g(value), b(value), a(1.0f) {}
    __device__ HIPRTColor(float _r, float _g, float _b) : r(_r), g(_g), b(_b), a(1.0f) {}
    __device__ HIPRTColor(float _r, float _g, float _b, float _a) : r(_r), g(_g), b(_b), a(_a) {}

    float r, g, b, a;
};

#ifdef __KERNELCC__
__device__ float luminance(const HIPRTColor& color)
{
    return 0.3086f * color.r + 0.6094f * color.g + 0.0820f * color.b;
}

__device__ float max(const HIPRTColor& color)
{
    return RT_MAX(color.r, RT_MAX(color.g, RT_MAX(color.b, float(0))));
}

__device__ HIPRTColor operator+ (const HIPRTColor& a, const HIPRTColor& b)
{
    return HIPRTColor{ a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a };
}

__device__ HIPRTColor operator- (const HIPRTColor& c)
{
    return HIPRTColor{ -c.r, -c.g, -c.b, -c.a };
}

__device__ HIPRTColor operator- (const HIPRTColor& a, const HIPRTColor& b)
{
    return a + (-b);
}

__device__ HIPRTColor operator* (const HIPRTColor& a, const HIPRTColor& b)
{
    return HIPRTColor{ a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a };
}

__device__ HIPRTColor operator* (const float k, const HIPRTColor& c)
{
    return HIPRTColor{ c.r * k, c.g * k, c.b * k, c.a * k };
}

__device__ HIPRTColor operator* (const HIPRTColor& c, const float k)
{
    return k * c;
}

__device__ HIPRTColor operator/ (const HIPRTColor& a, const HIPRTColor& b)
{
    return HIPRTColor{ a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a };
}

__device__ HIPRTColor operator/ (const float k, const HIPRTColor& c)
{
    return HIPRTColor{ k / c.r, k / c.g, k / c.b, k / c.a };
}

__device__ HIPRTColor operator/ (const HIPRTColor& c, const float k)
{
    float kk = 1 / k;
    return kk * c;
}

__device__ HIPRTColor exp(const HIPRTColor& col)
{
    return HIPRTColor{ expf(col.r), expf(col.g), expf(col.b), col.a };
}

__device__ HIPRTColor pow(const HIPRTColor& col, float k)
{
    return HIPRTColor{ powf(col.r, k), powf(col.g, k), powf(col.b, k), col.a };
}
#endif

#endif
