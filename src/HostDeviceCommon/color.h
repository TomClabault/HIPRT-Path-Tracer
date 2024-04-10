#ifndef COLOR_H
#define COLOR_H

#include <hiprt/hiprt_device.h>

#ifndef __KERNELCC__
#define __prefix__ inline
#else
#define __prefix__ __device__
#endif

struct Color
{
    __prefix__ Color() : r(0.0f), g(0.0f), b(0.0f) {}
    __prefix__ explicit Color(float value) : r(value), g(value), b(value) {}
    __prefix__ Color(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
    __prefix__ Color(hiprtFloat3 vec) : r(vec.x), g(vec.y), b(vec.z) {}

    __prefix__ Color& operator+=(const Color& other)
    {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    __prefix__ Color& operator-=(const Color& other)
    {
        r -= other.r;
        g -= other.g;
        b -= other.b;

        return *this;
    }

    __prefix__ Color& operator*=(const Color& other)
    {
        r *= other.r;
        g *= other.g;
        b *= other.b;

        return *this;
    }

    __prefix__ Color& operator/=(const Color& other)
    {
        r /= other.r;
        g /= other.g;
        b /= other.b;

        return *this;
    }

    __prefix__ Color& operator/=(float k)
    {
        r /= k;
        g /= k;
        b /= k;

        return *this;
    }

    __prefix__ bool operator!=(const Color& other)
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
__prefix__ float max(const Color& color)
{
    return _COLOR_MAX(color.r, _COLOR_MAX(color.g, _COLOR_MAX(color.b, float(0))));
}

__prefix__ Color operator+ (const Color& a, const Color& b)
{
    return Color{ a.r + b.r, a.g + b.g, a.b + b.b };
}

__prefix__ Color operator- (const Color& c)
{
    return Color{ -c.r, -c.g, -c.b };
}

__prefix__ Color operator- (const Color& a, const Color& b)
{
    return a + (-b);
}

__prefix__ Color operator* (const Color& a, const Color& b)
{
    return Color{ a.r * b.r, a.g * b.g, a.b * b.b};
}

__prefix__ Color operator* (const float k, const Color& c)
{
    return Color{ c.r * k, c.g * k, c.b * k };
}

__prefix__ Color operator* (const Color& c, const float k)
{
    return k * c;
}

__prefix__ Color operator/ (const Color& a, const Color& b)
{
    return Color{ a.r / b.r, a.g / b.g, a.b / b.b };
}

__prefix__ Color operator/ (const float k, const Color& c)
{
    return Color{ k / c.r, k / c.g, k / c.b };
}

__prefix__ Color operator/ (const Color& c, const float k)
{
    float kk = 1 / k;
    return kk * c;
}

__prefix__ Color sqrt(const Color& col)
{
    return Color{ sqrt(col.r), sqrt(col.g), sqrt(col.b) };
}

__prefix__ Color exp(const Color& col)
{
    return Color{ expf(col.r), expf(col.g), expf(col.b) };
}

__prefix__ Color pow(const Color& col, float k)
{
    return Color{ powf(col.r, k), powf(col.g, k), powf(col.b, k) };
}

#ifndef __KERNELCC__ // Only defining this on the CPU side
inline std::ostream& operator<<(std::ostream& os, const Color& color)
{
    os << "[" << color.r << ", " << color.g << ", " << color.b << "]";

    return os;
}
#endif

#endif