/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef COLOR_H
#define COLOR_H

#include "HostDeviceCommon/Math.h"

#ifndef __KERNELCC__
#define __prefix__ inline
#else
#define __prefix__ __device__
#endif

struct ColorRGB
{
    __prefix__ ColorRGB() : r(0.0f), g(0.0f), b(0.0f) {}
    __prefix__ explicit ColorRGB(float value) : r(value), g(value), b(value) {}
    __prefix__ ColorRGB(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
    __prefix__ ColorRGB(float3 vec) : r(vec.x), g(vec.y), b(vec.z) {}

    __prefix__ ColorRGB& operator+=(const ColorRGB& other)
    {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    __prefix__ ColorRGB& operator-=(const ColorRGB& other)
    {
        r -= other.r;
        g -= other.g;
        b -= other.b;

        return *this;
    }

    __prefix__ ColorRGB& operator*=(const ColorRGB& other)
    {
        r *= other.r;
        g *= other.g;
        b *= other.b;

        return *this;
    }

    __prefix__ ColorRGB& operator/=(const ColorRGB& other)
    {
        r /= other.r;
        g /= other.g;
        b /= other.b;

        return *this;
    }

    __prefix__ ColorRGB& operator/=(float k)
    {
        r /= k;
        g /= k;
        b /= k;

        return *this;
    }

    __prefix__ bool operator!=(const ColorRGB& other)
    {
        return r != other.r || g != other.g || b != other.b;
    }

    __prefix__ float luminance() const
    {
        return 0.3086f * r + 0.6094f * g + 0.0820f * b;
    }

    float r, g, b;
};

#define _COLOR_MAX( a, b ) ( ( ( b ) > ( a ) ) ? ( b ) : ( a ) )
__prefix__ float max(const ColorRGB& color)
{
    return _COLOR_MAX(color.r, _COLOR_MAX(color.g, _COLOR_MAX(color.b, float(0))));
}

__prefix__ ColorRGB operator+ (const ColorRGB& a, const ColorRGB& b)
{
    return ColorRGB{ a.r + b.r, a.g + b.g, a.b + b.b };
}

__prefix__ ColorRGB operator- (const ColorRGB& c)
{
    return ColorRGB{ -c.r, -c.g, -c.b };
}

__prefix__ ColorRGB operator- (const ColorRGB& a, const ColorRGB& b)
{
    return a + (-b);
}

__prefix__ ColorRGB operator* (const ColorRGB& a, const ColorRGB& b)
{
    return ColorRGB{ a.r * b.r, a.g * b.g, a.b * b.b};
}

__prefix__ ColorRGB operator* (const float k, const ColorRGB& c)
{
    return ColorRGB{ c.r * k, c.g * k, c.b * k };
}

__prefix__ ColorRGB operator* (const ColorRGB& c, const float k)
{
    return k * c;
}

__prefix__ ColorRGB operator/ (const ColorRGB& a, const ColorRGB& b)
{
    return ColorRGB{ a.r / b.r, a.g / b.g, a.b / b.b };
}

__prefix__ ColorRGB operator/ (const float k, const ColorRGB& c)
{
    return ColorRGB{ k / c.r, k / c.g, k / c.b };
}

__prefix__ ColorRGB operator/ (const ColorRGB& c, const float k)
{
    float kk = 1 / k;
    return kk * c;
}

__prefix__ ColorRGB sqrt(const ColorRGB& col)
{
    return ColorRGB{ sqrt(col.r), sqrt(col.g), sqrt(col.b) };
}

__prefix__ ColorRGB exp(const ColorRGB& col)
{
    return ColorRGB{ expf(col.r), expf(col.g), expf(col.b) };
}

__prefix__ ColorRGB pow(const ColorRGB& col, float k)
{
    return ColorRGB{ powf(col.r, k), powf(col.g, k), powf(col.b, k) };
}

#ifndef __KERNELCC__ // Only defining this on the CPU side
inline std::ostream& operator<<(std::ostream& os, const ColorRGB& color)
{
    os << "[" << color.r << ", " << color.g << ", " << color.b << "]";

    return os;
}
#endif

#endif