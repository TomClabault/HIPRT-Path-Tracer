/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_COLOR_H
#define HOST_DEVICE_COMMON_COLOR_H

#include "HostDeviceCommon/Math.h"

struct ColorRGBA32F
{
    HIPRT_HOST_DEVICE ColorRGBA32F() { r = 0.0f; g = 0.0f; b = 0.0f; a = 1.0f; }
    HIPRT_HOST_DEVICE explicit ColorRGBA32F(float value) { r = value; g = value; b = value; a = 1.0f; }
    HIPRT_HOST_DEVICE ColorRGBA32F(float _r, float _g, float _b, float _a) { r = _r; g = _g; b = _b; a = _a; }
    HIPRT_HOST_DEVICE explicit ColorRGBA32F(float4 vec) { r = vec.x; g = vec.y; b = vec.z; a = vec.w; }

    HIPRT_HOST_DEVICE void operator+=(const ColorRGBA32F& other) { r += other.r; g += other.g; b += other.b; a += other.a; }
    HIPRT_HOST_DEVICE void operator-=(const ColorRGBA32F& other) { r -= other.r; g -= other.g; b -= other.b; a -= other.a; }
    HIPRT_HOST_DEVICE void operator*=(const ColorRGBA32F& other) { r *= other.r; g *= other.g; b *= other.b; a *= other.a; }
    HIPRT_HOST_DEVICE void operator*=(float k) { r *= k; g *= k; b *= k; a *= k; }
    HIPRT_HOST_DEVICE void operator/=(const ColorRGBA32F& other) { r /= other.r; g /= other.g; b /= other.b; a /= other.a; }
    HIPRT_HOST_DEVICE void operator/=(float k) { r /= k; g /= k; b /= k; a /= k; }
    HIPRT_HOST_DEVICE bool operator!=(const ColorRGBA32F& other) { return r != other.r || g != other.g || b != other.g || a != other.a; }

    HIPRT_HOST_DEVICE float length() const { return sqrtf(this->length2()); }
    HIPRT_HOST_DEVICE float length2() const { return r * r + g * g + b * b + a * a; }
    HIPRT_HOST_DEVICE float luminance() const { return 0.3086f * r + 0.6094f * g + 0.0820f * b; }
    HIPRT_HOST_DEVICE void clamp(float min, float max) { r = hippt::clamp(min, max, r); g = hippt::clamp(min, max, g); b = hippt::clamp(min, max, b); a = hippt::clamp(min, max, a); }
    HIPRT_HOST_DEVICE bool has_NaN() const { return hippt::is_nan(r) || hippt::is_nan(g) || hippt::is_nan(b) || hippt::is_nan(a); }
    HIPRT_HOST_DEVICE bool is_black() const { return !(r > 0.0f || g > 0.0f || b > 0.0f); }
    HIPRT_HOST_DEVICE bool is_white() const { return r == 1.0f && g == 1.0f && b == 1.0f; }


    HIPRT_HOST_DEVICE float max_component() const { return hippt::max(r, hippt::max(g, hippt::max(b, a))); }
    HIPRT_HOST_DEVICE ColorRGBA32F normalized() const { float length = sqrtf(r * r + g * g + b * b); return ColorRGBA32F(r / length, g / length, b / length, /* not normalizing alpha */ a); }

    HIPRT_HOST_DEVICE static ColorRGBA32F max(const ColorRGBA32F& a, const ColorRGBA32F& b) { return ColorRGBA32F(hippt::max(a.r, b.r), hippt::max(a.g, b.g), hippt::max(a.b, b.b), hippt::max(a.a, b.a)); }
    HIPRT_HOST_DEVICE static ColorRGBA32F min(const ColorRGBA32F& a, const ColorRGBA32F& b) { return ColorRGBA32F(hippt::min(a.r, b.r), hippt::min(a.g, b.g), hippt::min(a.b, b.b), hippt::min(a.a, b.a)); }

    HIPRT_HOST_DEVICE float& operator[](int index) { return *(&r + index); }
    HIPRT_HOST_DEVICE float operator[](int index) const { return *(&r + index); }

    float r, g, b, a;
};

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator+ (const ColorRGBA32F& a, const ColorRGBA32F& b) { return ColorRGBA32F(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator- (const ColorRGBA32F& c) { return ColorRGBA32F(-c.r, -c.g, -c.b, c.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator- (const ColorRGBA32F& a, const ColorRGBA32F& b) { return ColorRGBA32F(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator* (const ColorRGBA32F& a, const ColorRGBA32F& b) { return ColorRGBA32F(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator* (const float k, const ColorRGBA32F& c) { return ColorRGBA32F(c.r * k, c.g * k, c.b * k, c.a * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator* (const ColorRGBA32F& c, const float k) { return ColorRGBA32F(c.r * k, c.g * k, c.b * k, c.a * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator/ (const ColorRGBA32F& a, const ColorRGBA32F& b) { return ColorRGBA32F(a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator/ (const float k, const ColorRGBA32F& c) { return ColorRGBA32F(k / c.r, k / c.g, k / c.b, k / c.a); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F operator/ (const ColorRGBA32F& c, const float k) { return ColorRGBA32F(c.r / k, c.g / k, c.b / k, c.a / k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F sqrt(const ColorRGBA32F& col) { return ColorRGBA32F(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b), sqrtf(col.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F exp(const ColorRGBA32F& col) { return ColorRGBA32F(expf(col.r), expf(col.g), expf(col.b), expf(col.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F log(const ColorRGBA32F& col) { return ColorRGBA32F(logf(col.r), logf(col.g), logf(col.b), logf(col.a)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGBA32F pow(const ColorRGBA32F& col, float k) { return ColorRGBA32F(powf(col.r, k), powf(col.g, k), powf(col.b, k), powf(col.a, k)); }

struct ColorRGB32F
{
    HIPRT_HOST_DEVICE ColorRGB32F() { r = 0.0f; g = 0.0f; b = 0.0f; }
    HIPRT_HOST_DEVICE explicit ColorRGB32F(float value) { r = value; g = value; b = value; }
    HIPRT_HOST_DEVICE ColorRGB32F(float _r, float _g, float _b) { r = _r; g = _g; b = _b; }
    HIPRT_HOST_DEVICE explicit ColorRGB32F(float3 vec) { r = vec.x; g = vec.y; b = vec.z; }
    // W component of float4 is dropped
    HIPRT_HOST_DEVICE explicit ColorRGB32F(float4 vec) { r = vec.x; g = vec.y; b = vec.z; }
    // This constructor drops the alpha channel
    HIPRT_HOST_DEVICE explicit ColorRGB32F(const ColorRGBA32F& rgba) { r = rgba.r; g = rgba.g; b = rgba.b; }

    HIPRT_HOST_DEVICE void operator+=(const ColorRGB32F& other) { r += other.r; g += other.g; b += other.b; }
    HIPRT_HOST_DEVICE void operator-=(const ColorRGB32F& other) { r -= other.r; g -= other.g; b -= other.b; }
    HIPRT_HOST_DEVICE void operator*=(const ColorRGB32F& other) { r *= other.r; g *= other.g; b *= other.b; }
    HIPRT_HOST_DEVICE void operator*=(float k) { r *= k; g *= k; b *= k; }
    HIPRT_HOST_DEVICE void operator/=(const ColorRGB32F& other) { r /= other.r; g /= other.g; b /= other.b; }
    HIPRT_HOST_DEVICE void operator/=(float k) { r /= k; g /= k; b /= k; }
    HIPRT_HOST_DEVICE bool operator!=(const ColorRGB32F& other) { return r != other.r || g != other.g || b != other.g; }

    HIPRT_HOST_DEVICE float length() const { return sqrtf(this->length2()); }
    HIPRT_HOST_DEVICE float length2() const { return r * r + g * g + b * b; }
    HIPRT_HOST_DEVICE float luminance() const { return 0.3086f * r + 0.6094f * g + 0.0820f * b; }
    HIPRT_HOST_DEVICE void clamp(float min, float max) { r = hippt::clamp(min, max, r); g = hippt::clamp(min, max, g); b = hippt::clamp(min, max, b); }
    HIPRT_HOST_DEVICE bool has_NaN() const { return hippt::is_nan(r) || hippt::is_nan(g) || hippt::is_nan(b); }
    HIPRT_HOST_DEVICE bool is_black() const { return !(r > 0.0f || g > 0.0f || b > 0.0f); }
    HIPRT_HOST_DEVICE bool is_white() const { return r == 1.0f && g == 1.0f && b == 1.0f; }

    HIPRT_HOST_DEVICE float max_component() const { return hippt::max(r, hippt::max(g, b)); }
    HIPRT_HOST_DEVICE ColorRGB32F normalized() const { float length = sqrtf(r * r + g * g + b * b); return ColorRGB32F(r / length, g / length, b / length); }

    HIPRT_HOST_DEVICE static ColorRGB32F max(const ColorRGB32F& a, const ColorRGB32F& b) { return ColorRGB32F(hippt::max(a.r, b.r), hippt::max(a.g, b.g), hippt::max(a.b, b.b)); }
    HIPRT_HOST_DEVICE static ColorRGB32F min(const ColorRGB32F& a, const ColorRGB32F& b) { return ColorRGB32F(hippt::min(a.r, b.r), hippt::min(a.g, b.g), hippt::min(a.b, b.b)); }

    HIPRT_HOST_DEVICE float& operator[](int index) { return *(&r + index); }
    HIPRT_HOST_DEVICE float operator[](int index) const { return *(&r + index); }

    float r, g, b;
};

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator+ (const ColorRGB32F& a, const ColorRGB32F& b) { return ColorRGB32F(a.r + b.r, a.g + b.g, a.b + b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator- (const ColorRGB32F& c) { return ColorRGB32F(-c.r, -c.g, -c.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator- (const ColorRGB32F& a, const ColorRGB32F& b) { return ColorRGB32F(a.r - b.r, a.g - b.g, a.b - b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator* (const ColorRGB32F& a, const ColorRGB32F& b) { return ColorRGB32F(a.r * b.r, a.g * b.g, a.b * b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator* (const float k, const ColorRGB32F& c) { return ColorRGB32F(c.r * k, c.g * k, c.b * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator* (const ColorRGB32F& c, const float k) { return ColorRGB32F(c.r * k, c.g * k, c.b * k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator/ (const ColorRGB32F& a, const ColorRGB32F& b) { return ColorRGB32F(a.r / b.r, a.g / b.g, a.b / b.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator/ (const float k, const ColorRGB32F& c) { return ColorRGB32F(k / c.r, k / c.g, k / c.b); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F operator/ (const ColorRGB32F& c, const float k) { return ColorRGB32F(c.r / k, c.g / k, c.b / k); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sqrt(const ColorRGB32F& col) { return ColorRGB32F(sqrtf(col.r), sqrtf(col.g), sqrtf(col.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F exp(const ColorRGB32F& col) { return ColorRGB32F(expf(col.r), expf(col.g), expf(col.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F log(const ColorRGB32F& col) { return ColorRGB32F(logf(col.r), logf(col.g), logf(col.b)); }
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F pow(const ColorRGB32F& col, float k) { return ColorRGB32F(powf(col.r, k), powf(col.g, k), powf(col.b, k)); }

#ifndef __KERNELCC__
inline std::ostream& operator <<(std::ostream& os, const ColorRGB32F& color)
{
    os << color.r << ", " << color.g << ", " << color.b;

    return os;
}

inline std::ostream& operator <<(std::ostream& os, const ColorRGBA32F& color)
{
    os << color.r << ", " << color.g << ", " << color.b << ", " << color.a;

    return os;
}
#endif

#endif
