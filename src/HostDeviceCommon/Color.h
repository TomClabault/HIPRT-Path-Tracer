/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef COLOR_H
#define COLOR_H

#include "HostDeviceCommon/Math.h"

struct ColorRGB
{
    HIPRT_HOST_DEVICE ColorRGB() { r = 0.0f; g = 0.0f; b = 0.0f; }
    HIPRT_HOST_DEVICE explicit ColorRGB(float value) { r = value; g = value; b = value; }
    HIPRT_HOST_DEVICE ColorRGB(float _r, float _g, float _b) { r = _r; g = _g; b = _b; }
    HIPRT_HOST_DEVICE ColorRGB(float3 vec) { r = vec.x; g = vec.y; b = vec.z; }

    HIPRT_HOST_DEVICE void operator+=(const ColorRGB& other) { r += other.r; g += other.g; b += other.b; }
    HIPRT_HOST_DEVICE void operator-=(const ColorRGB& other) { r -= other.r; g -= other.g; b -= other.b; }
    HIPRT_HOST_DEVICE void operator*=(const ColorRGB& other) { r *= other.r; g *= other.g; b *= other.b; }
    HIPRT_HOST_DEVICE void operator*=(float k) { r *= k; g *= k; b *= k; }
    HIPRT_HOST_DEVICE void operator/=(const ColorRGB& other) { r /= other.r; g /= other.g; b /= other.b; }
    HIPRT_HOST_DEVICE void operator/=(float k) { r /= k; g /= k; b /= k; }
    HIPRT_HOST_DEVICE bool operator!=(const ColorRGB& other) { return r != other.r || g != other.g || b != other.g; }

    HIPRT_HOST_DEVICE float length() const { return sqrtf(this->length2()); }
    HIPRT_HOST_DEVICE float length2() const { return r * r + g * g + b * b; }
    HIPRT_HOST_DEVICE float luminance() const { return 0.3086f * r + 0.6094f * g + 0.0820f * b; }
    HIPRT_HOST_DEVICE void clamp(float min, float max) { r = hippt::clamp(min, max, r); g = hippt::clamp(min, max, g); b = hippt::clamp(min, max, b); }
    HIPRT_HOST_DEVICE bool has_NaN() { return hippt::isNaN(r) || hippt::isNaN(g) || hippt::isNaN(b); }

    float r, g, b;
};

//HIPRT_HOST_DEVICE HIPRT_INLINE float max(const ColorRGB& color) { return hippt::max(color.r, hippt::max(color.g, color.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB max(const ColorRGB& a, const ColorRGB& b) { return ColorRGB(hippt::max(a.r, b.r), hippt::max(a.g, b.g), hippt::max(a.b, b.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator+ (const ColorRGB& a, const ColorRGB& b) { return ColorRGB(a.r + b.r, a.g + b.g, a.b + b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator- (const ColorRGB& c) { return ColorRGB(-c.r, -c.g, -c.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator- (const ColorRGB& a, const ColorRGB& b) { return ColorRGB(a.r - b.r, a.g - b.g, a.b - b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator* (const ColorRGB& a, const ColorRGB& b) { return ColorRGB(a.r * b.r, a.g * b.g, a.b * b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator* (const float k, const ColorRGB& c) { return ColorRGB(c.r * k, c.g * k, c.b * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator* (const ColorRGB& c, const float k) { return ColorRGB(c.r * k, c.g * k, c.b * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator/ (const ColorRGB& a, const ColorRGB& b) { return ColorRGB(a.r * b.r, a.g / b.g, a.b / b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator/ (const float k, const ColorRGB& c) { return ColorRGB(k / c.r, k / c.g, k / c.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB operator/ (const ColorRGB& c, const float k) { return ColorRGB(c.r / k, c.g / k, c.b / k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sqrt(const ColorRGB& col) { return ColorRGB(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB exp(const ColorRGB& col) { return ColorRGB(expf(col.r), expf(col.g), expf(col.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB log(const ColorRGB& col) { return ColorRGB(logf(col.r), logf(col.g), logf(col.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB pow(const ColorRGB& col, float k) { return ColorRGB(powf(col.r, k), powf(col.g, k), powf(col.b, k)); }

struct ColorRGBA
{
    HIPRT_HOST_DEVICE ColorRGBA() { r = 0.0f; g = 0.0f; b = 0.0f; a = 1.0f; }
    HIPRT_HOST_DEVICE explicit ColorRGBA(float value) { r = value; g = value; b = value; a = 1.0f; }
    HIPRT_HOST_DEVICE ColorRGBA(float _r, float _g, float _b, float _a) { r = _r; g = _g; b = _b; a = _a; }
    HIPRT_HOST_DEVICE ColorRGBA(float4 vec) { r = vec.x; g = vec.y; b = vec.z; a = vec.w; }

    HIPRT_HOST_DEVICE void operator+=(const ColorRGBA& other) { r += other.r; g += other.g; b += other.b; a += other.a; }
    HIPRT_HOST_DEVICE void operator-=(const ColorRGBA& other) { r -= other.r; g -= other.g; b -= other.b; a -= other.a; }
    HIPRT_HOST_DEVICE void operator*=(const ColorRGBA& other) { r *= other.r; g *= other.g; b *= other.b; a *= other.a; }
    HIPRT_HOST_DEVICE void operator*=(float k) { r *= k; g *= k; b *= k; a *= k; }
    HIPRT_HOST_DEVICE void operator/=(const ColorRGBA& other) { r /= other.r; g /= other.g; b /= other.b; a /= other.a; }
    HIPRT_HOST_DEVICE void operator/=(float k) { r /= k; g /= k; b /= k; a /= k; }
    HIPRT_HOST_DEVICE bool operator!=(const ColorRGBA& other) { return r != other.r || g != other.g || b != other.g || a != other.a; }

    HIPRT_HOST_DEVICE float length() const { return sqrtf(this->length2()); }
    HIPRT_HOST_DEVICE float length2() const { return r * r + g * g + b * b + a * a; }
    HIPRT_HOST_DEVICE float luminance() const { return 0.3086f * r + 0.6094f * g + 0.0820f * b; }
    HIPRT_HOST_DEVICE void clamp(float min, float max) { r = hippt::clamp(min, max, r); g = hippt::clamp(min, max, g); b = hippt::clamp(min, max, b); a = hippt::clamp(min, max, a); }
    HIPRT_HOST_DEVICE bool has_NaN() { return hippt::isNaN(r) || hippt::isNaN(g) || hippt::isNaN(b) || hippt::isNaN(a); }

    float r, g, b, a;
};

//HIPRT_HOST_DEVICE HIPRT_INLINE float max(const ColorRGBA& color) { return hippt::max(color.r, hippt::max(color.g, hippt::max(color.b, color.a))); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA max(const ColorRGBA& a, const ColorRGBA& b) { return ColorRGBA(hippt::max(a.r, b.r), hippt::max(a.g, b.g), hippt::max(a.b, b.b), hippt::max(a.a, b.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator+ (const ColorRGBA& a, const ColorRGBA& b) { return ColorRGBA(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator- (const ColorRGBA& c) { return ColorRGBA(-c.r, -c.g, -c.b, c.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator- (const ColorRGBA& a, const ColorRGBA& b) { return ColorRGBA(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator* (const ColorRGBA& a, const ColorRGBA& b) { return ColorRGBA(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator* (const float k, const ColorRGBA& c) { return ColorRGBA(c.r * k, c.g * k, c.b * k, c.a * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator* (const ColorRGBA& c, const float k) { return ColorRGBA(c.r * k, c.g * k, c.b * k, c.a * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator/ (const ColorRGBA& a, const ColorRGBA& b) { return ColorRGBA(a.r * b.r, a.g / b.g, a.b / b.b, a.a / b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator/ (const float k, const ColorRGBA& c) { return ColorRGBA(k / c.r, k / c.g, k / c.b, k / c.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA operator/ (const ColorRGBA& c, const float k) { return ColorRGBA(c.r / k, c.g / k, c.b / k, c.a / k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA sqrt(const ColorRGBA& col) { return ColorRGBA(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b), sqrtf(col.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA exp(const ColorRGBA& col) { return ColorRGBA(expf(col.r), expf(col.g), expf(col.b), expf(col.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA log(const ColorRGBA& col) { return ColorRGBA(logf(col.r), logf(col.g), logf(col.b), logf(col.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA pow(const ColorRGBA& col, float k) { return ColorRGBA(powf(col.r, k), powf(col.g, k), powf(col.b, k), powf(col.a, k)); }

#ifndef __KERNELCC__
inline std::ostream& operator <<(std::ostream& os, const ColorRGB& color)
{
    os << color.r << ", " << color.g << ", " << color.b << std::endl;

    return os;
}

inline std::ostream& operator <<(std::ostream& os, const ColorRGBA& color)
{
    os << color.r << ", " << color.g << ", " << color.b << ", " << color.a << std::endl;

    return os;
}
#endif

#endif