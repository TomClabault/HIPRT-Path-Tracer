
#ifndef HIPRT_COLOR_H
#define HIPRT_COLOR_H

#include <hiprt/hiprt_device.h>

#include "Kernels/includes/hiprt_fix_vs.h"
#include "Kernels/includes/HIPRT_maths.h"

#ifndef __KERNELCC__
#define __prefix__ inline
#else
#define __prefix__ __device__
#endif

// TODO instead of duplicating the structure, it would be better to create a folder
// HostDeviceCommon containing the structures that are used both by the GPU and CPU renderer
struct HIPRTColor
{
    __prefix__ HIPRTColor() : r(0.0f), g(0.0f), b(0.0f) {}
    __prefix__ HIPRTColor(float value) : r(value), g(value), b(value) {}
    __prefix__ HIPRTColor(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
    __prefix__ HIPRTColor(hiprtFloat3 vec) : r(vec.x), g(vec.y), b(vec.z) {}

    __prefix__ HIPRTColor& operator+=(const HIPRTColor& other)
    {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    __prefix__ HIPRTColor& operator-=(const HIPRTColor& other)
    {
        r -= other.r;
        g -= other.g;
        b -= other.b;

        return *this;
    }

    __prefix__ HIPRTColor& operator*=(const HIPRTColor& other)
    {
        r *= other.r;
        g *= other.g;
        b *= other.b;

        return *this;
    }

    __prefix__ HIPRTColor& operator/=(const HIPRTColor& other)
    {
        r /= other.r;
        g /= other.g;
        b /= other.b;

        return *this;
    }

    __prefix__ bool operator!=(const HIPRTColor& other)
    {
        return r != other.r || g != other.g || b != other.b;
    }

    __prefix__ float luminance()
    {
        return 0.3086f * r + 0.6094f * g + 0.0820f * b;
    }

    float r, g, b, a = 0.0f;
};

__prefix__ float max(const HIPRTColor& color)
{
    return RT_MAX(color.r, RT_MAX(color.g, RT_MAX(color.b, float(0))));
}

__prefix__ HIPRTColor operator+ (const HIPRTColor& a, const HIPRTColor& b)
{
    return HIPRTColor{ a.r + b.r, a.g + b.g, a.b + b.b };
}

__prefix__ HIPRTColor operator- (const HIPRTColor& c)
{
    return HIPRTColor{ -c.r, -c.g, -c.b };
}

__prefix__ HIPRTColor operator- (const HIPRTColor& a, const HIPRTColor& b)
{
    return a + (-b);
}

__prefix__ HIPRTColor operator* (const HIPRTColor& a, const HIPRTColor& b)
{
    return HIPRTColor{ a.r * b.r, a.g * b.g, a.b * b.b};
}

__prefix__ HIPRTColor operator* (const float k, const HIPRTColor& c)
{
    return HIPRTColor{ c.r * k, c.g * k, c.b * k };
}

__prefix__ HIPRTColor operator* (const HIPRTColor& c, const float k)
{
    return k * c;
}

__prefix__ HIPRTColor operator/ (const HIPRTColor& a, const HIPRTColor& b)
{
    return HIPRTColor{ a.r / b.r, a.g / b.g, a.b / b.b };
}

__prefix__ HIPRTColor operator/ (const float k, const HIPRTColor& c)
{
    return HIPRTColor{ k / c.r, k / c.g, k / c.b };
}

__prefix__ HIPRTColor operator/ (const HIPRTColor& c, const float k)
{
    float kk = 1 / k;
    return kk * c;
}

__prefix__ HIPRTColor exp(const HIPRTColor& col)
{
    return HIPRTColor{ expf(col.r), expf(col.g), expf(col.b) };
}

__prefix__ HIPRTColor pow(const HIPRTColor& col, float k)
{
    return HIPRTColor{ powf(col.r, k), powf(col.g, k), powf(col.b, k) };
}

#endif
