/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_COLOR_H
#define HOST_DEVICE_COMMON_COLOR_H

#include "Device/includes/Hash.h"
#include "HostDeviceCommon/Maths/Math.h"

struct ColorRGBA32F
{
	HIPRT_DEVICE ColorRGBA32F()
	{
		r = 0.0f;
		g = 0.0f;
		b = 0.0f;
		a = 1.0f;
	}

	HIPRT_DEVICE explicit ColorRGBA32F(float value)
	{
		r = value;
		g = value;
		b = value;
		a = 1.0f;
	}

	HIPRT_DEVICE ColorRGBA32F(float _r, float _g, float _b, float _a)
	{
		r = _r;
		g = _g;
		b = _b;
		a = _a;
	}

	HIPRT_DEVICE explicit ColorRGBA32F(float4_t vec)
	{
		r = vec.x;
		g = vec.y;
		b = vec.z;
		a = vec.w;
	}

	HIPRT_DEVICE void operator+=(const ColorRGBA32F& other)
	{
		r += other.r;
		g += other.g;
		b += other.b;
		a += other.a;
	}

	HIPRT_DEVICE void operator-=(const ColorRGBA32F& other)
	{
		r -= other.r;
		g -= other.g;
		b -= other.b;
		a -= other.a;
	}

	HIPRT_DEVICE void operator*=(const ColorRGBA32F& other)
	{
		r *= other.r;
		g *= other.g;
		b *= other.b;
		a *= other.a;
	}

	HIPRT_DEVICE void operator*=(float k)
	{
		r *= k;
		g *= k;
		b *= k;
		a *= k;
	}

	HIPRT_DEVICE void operator/=(const ColorRGBA32F& other)
	{
		r /= other.r;
		g /= other.g;
		b /= other.b;
		a /= other.a;
	}

	HIPRT_DEVICE void operator/=(float k)
	{
		r /= k;
		g /= k;
		b /= k;
		a /= k;
	}

	HIPRT_DEVICE bool operator!=(const ColorRGBA32F& other)
	{
		return r != other.r || g != other.g || b != other.g || a != other.a;
	}

	HIPRT_DEVICE float length() const
	{
		return hippt::sqrt(this->length2());
	}

	HIPRT_DEVICE float length2() const
	{
		return r * r + g * g + b * b + a * a;
	}

	HIPRT_DEVICE float luminance() const
	{
		return 0.3086f * r + 0.6094f * g + 0.0820f * b;
	}

	HIPRT_DEVICE void clamp(float min, float max)
	{
		r = hippt::clamp(min, max, r);
		g = hippt::clamp(min, max, g);
		b = hippt::clamp(min, max, b);
		a = hippt::clamp(min, max, a);
	}

	HIPRT_DEVICE ColorRGBA32F clamped(float min, float max)
	{
		return ColorRGBA32F(hippt::clamp(min, max, r), g = hippt::clamp(min, max, g), b = hippt::clamp(min, max, b), a = hippt::clamp(min, max, a));
	}

	HIPRT_DEVICE bool has_nan() const
	{
		return hippt::is_nan(r) || hippt::is_nan(g) || hippt::is_nan(b) || hippt::is_nan(a);
	}

	HIPRT_DEVICE bool has_inf() const
	{
		return hippt::is_inf(r) || hippt::is_inf(g) || hippt::is_inf(b) || hippt::is_inf(a);
	}

	HIPRT_DEVICE bool has_nan_or_inf() const
	{
		return has_nan() || has_inf();
	}

	HIPRT_DEVICE bool is_black() const
	{
		return !(r > 0.0f || g > 0.0f || b > 0.0f);
	}

	HIPRT_DEVICE bool is_white() const
	{
		return r == 1.0f && g == 1.0f && b == 1.0f;
	}

	HIPRT_DEVICE float max_component() const
	{
		return hippt::max(r, hippt::max(g, hippt::max(b, a)));
	}

	HIPRT_DEVICE float min_component() const
	{
		return hippt::min(r, hippt::min(g, hippt::min(b, a)));
	}

	HIPRT_DEVICE ColorRGBA32F normalized() const
	{
		float length = hippt::sqrt(r * r + g * g + b * b);
		return ColorRGBA32F(r / length, g / length, b / length, /* not normalizing alpha */ a);
	}

	HIPRT_DEVICE ColorRGBA32F abs()
	{
		return ColorRGBA32F(hippt::abs(this->r), hippt::abs(this->g), hippt::abs(this->b), hippt::abs(this->a));
	}

	HIPRT_DEVICE void max(const ColorRGBA32F& maxer)
	{
		this->r = hippt::max(this->r, maxer.r);
		this->g = hippt::max(this->g, maxer.g);
		this->b = hippt::max(this->b, maxer.b);
		this->a = hippt::max(this->a, maxer.a);
	}

	HIPRT_DEVICE ColorRGBA32F maxed(const ColorRGBA32F& maxer)
	{
		return ColorRGBA32F(hippt::max(this->r, maxer.r), hippt::max(this->g, maxer.g), hippt::max(this->b, maxer.b), hippt::max(this->a, maxer.a));
	}

	HIPRT_DEVICE static ColorRGBA32F max(const ColorRGBA32F& a, const ColorRGBA32F& b)
	{
		return ColorRGBA32F(hippt::max(a.r, b.r), hippt::max(a.g, b.g), hippt::max(a.b, b.b), hippt::max(a.a, b.a));
	}

	HIPRT_DEVICE static ColorRGBA32F min(const ColorRGBA32F& a, const ColorRGBA32F& b)
	{
		return ColorRGBA32F(hippt::min(a.r, b.r), hippt::min(a.g, b.g), hippt::min(a.b, b.b), hippt::min(a.a, b.a));
	}

	HIPRT_DEVICE float& operator[](int index)
	{
		return *(&r + index);
	}

	HIPRT_DEVICE float operator[](int index) const
	{
		return *(&r + index);
	}

	float r, g, b, a;
};

HIPRT_DEVICE static ColorRGBA32F operator+(const ColorRGBA32F& a, const ColorRGBA32F& b)
{
	return ColorRGBA32F(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a);
}

HIPRT_DEVICE static ColorRGBA32F operator-(const ColorRGBA32F& c)
{
	return ColorRGBA32F(-c.r, -c.g, -c.b, c.a);
}

HIPRT_DEVICE static ColorRGBA32F operator-(const ColorRGBA32F& a, const ColorRGBA32F& b)
{
	return ColorRGBA32F(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a);
}

HIPRT_DEVICE static ColorRGBA32F operator*(const ColorRGBA32F& a, const ColorRGBA32F& b)
{
	return ColorRGBA32F(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a);
}

HIPRT_DEVICE static ColorRGBA32F operator*(const float k, const ColorRGBA32F& c)
{
	return ColorRGBA32F(c.r * k, c.g * k, c.b * k, c.a * k);
}

HIPRT_DEVICE static ColorRGBA32F operator*(const ColorRGBA32F& c, const float k)
{
	return ColorRGBA32F(c.r * k, c.g * k, c.b * k, c.a * k);
}

HIPRT_DEVICE static ColorRGBA32F operator/(const ColorRGBA32F& a, const ColorRGBA32F& b)
{
	return ColorRGBA32F(a.r / b.r, a.g / b.g, a.b / b.b, a.a / b.a);
}

HIPRT_DEVICE static ColorRGBA32F operator/(const float k, const ColorRGBA32F& c)
{
	return ColorRGBA32F(k / c.r, k / c.g, k / c.b, k / c.a);
}

HIPRT_DEVICE static ColorRGBA32F operator/(const ColorRGBA32F& c, const float k)
{
	return ColorRGBA32F(c.r / k, c.g / k, c.b / k, c.a / k);
}

HIPRT_DEVICE static ColorRGBA32F sqrt(const ColorRGBA32F& col)
{
	return ColorRGBA32F(hippt::sqrt(col.r), hippt::sqrt(col.g), hippt::sqrt(col.b), hippt::sqrt(col.a));
}

HIPRT_DEVICE static ColorRGBA32F exp(const ColorRGBA32F& col)
{
	return ColorRGBA32F(expf(col.r), expf(col.g), expf(col.b), expf(col.a));
}

HIPRT_DEVICE static ColorRGBA32F exp2(const ColorRGBA32F& col)
{
	return ColorRGBA32F(exp2f(col.r), exp2f(col.g), exp2f(col.b), exp2f(col.a));
}

HIPRT_DEVICE static ColorRGBA32F log(const ColorRGBA32F& col)
{
	return ColorRGBA32F(logf(col.r), logf(col.g), logf(col.b), logf(col.a));
}

HIPRT_DEVICE static ColorRGBA32F pow(const ColorRGBA32F& col, float k)
{
	return ColorRGBA32F(powf(col.r, k), powf(col.g, k), powf(col.b, k), powf(col.a, k));
}

HIPRT_DEVICE static ColorRGBA32F intrin_pow(ColorRGBA32F x, float y)
{
	return ColorRGBA32F(hippt::intrin_pow(x.r, y), hippt::intrin_pow(x.g, y), hippt::intrin_pow(x.b, y), hippt::intrin_pow(x.a, y));
}

HIPRT_DEVICE static ColorRGBA32F pow_2_2_fit(ColorRGBA32F x)
{
	return ColorRGBA32F(hippt::pow_2_2_fit(x.r), hippt::pow_2_2_fit(x.g), hippt::pow_2_2_fit(x.b), hippt::pow_2_2_fit(x.a));
}

struct ColorRGB32F
{
	HIPRT_DEVICE ColorRGB32F()
	{
		r = 0.0f;
		g = 0.0f;
		b = 0.0f;
	}

	HIPRT_DEVICE explicit ColorRGB32F(float value)
	{
		r = value;
		g = value;
		b = value;
	}

	HIPRT_DEVICE ColorRGB32F(float _r, float _g, float _b)
	{
		r = _r;
		g = _g;
		b = _b;
	}

	HIPRT_DEVICE explicit ColorRGB32F(float3_t vec)
	{
		r = vec.x;
		g = vec.y;
		b = vec.z;
	}
	// W component of float4_t is dropped
	HIPRT_DEVICE explicit ColorRGB32F(float4_t vec)
	{
		r = vec.x;
		g = vec.y;
		b = vec.z;
	}
	// This constructor drops the alpha channel
	HIPRT_DEVICE explicit ColorRGB32F(const ColorRGBA32F& rgba)
	{
		r = rgba.r;
		g = rgba.g;
		b = rgba.b;
	}

	HIPRT_DEVICE void operator+=(const ColorRGB32F& other)
	{
		r += other.r;
		g += other.g;
		b += other.b;
	}

	HIPRT_DEVICE void operator-=(const ColorRGB32F& other)
	{
		r -= other.r;
		g -= other.g;
		b -= other.b;
	}

	HIPRT_DEVICE void operator*=(const ColorRGB32F& other)
	{
		r *= other.r;
		g *= other.g;
		b *= other.b;
	}

	HIPRT_DEVICE void operator*=(float k)
	{
		r *= k;
		g *= k;
		b *= k;
	}

	HIPRT_DEVICE void operator/=(const ColorRGB32F& other)
	{
		r /= other.r;
		g /= other.g;
		b /= other.b;
	}

	HIPRT_DEVICE void operator/=(float k)
	{
		r /= k;
		g /= k;
		b /= k;
	}

	HIPRT_DEVICE bool operator!=(const ColorRGB32F& other)
	{
		return r != other.r || g != other.g || b != other.g;
	}

	HIPRT_DEVICE float length() const
	{
		return hippt::sqrt(this->length2());
	}

	HIPRT_DEVICE float length2() const
	{
		return r * r + g * g + b * b;
	}

	HIPRT_DEVICE float luminance() const
	{
		return 0.3086f * r + 0.6094f * g + 0.0820f * b;
	}

	HIPRT_DEVICE void clamp(float min, float max)
	{
		r = hippt::clamp(min, max, r);
		g = hippt::clamp(min, max, g);
		b = hippt::clamp(min, max, b);
	}

	HIPRT_DEVICE ColorRGB32F clamped(float min, float max)
	{
		return ColorRGB32F(hippt::clamp(min, max, r), g = hippt::clamp(min, max, g), b = hippt::clamp(min, max, b));
	}

	HIPRT_DEVICE bool has_nan() const
	{
		return hippt::is_nan(r) || hippt::is_nan(g) || hippt::is_nan(b);
	}

	HIPRT_DEVICE bool has_inf() const
	{
		return hippt::is_inf(r) || hippt::is_inf(g) || hippt::is_inf(b);
	}

	HIPRT_DEVICE bool has_nan_or_inf() const
	{
		return has_nan() || has_inf();
	}

	HIPRT_DEVICE bool is_black() const
	{
		return !(r > 0.0f || g > 0.0f || b > 0.0f);
	}

	HIPRT_DEVICE bool is_white() const
	{
		return r == 1.0f && g == 1.0f && b == 1.0f;
	}

	HIPRT_DEVICE float max_component() const
	{
		return hippt::max(r, hippt::max(g, b));
	}

	HIPRT_DEVICE float min_component() const
	{
		return hippt::min(r, hippt::min(g, b));
	}

	HIPRT_DEVICE ColorRGB32F normalized() const
	{
		float length = hippt::sqrt(r * r + g * g + b * b);
		return ColorRGB32F(r / length, g / length, b / length);
	}

	HIPRT_DEVICE ColorRGB32F abs()
	{
		return ColorRGB32F(hippt::abs(this->r), hippt::abs(this->g), hippt::abs(this->b));
	}

	HIPRT_DEVICE void max(const ColorRGB32F& maxer)
	{
		this->r = hippt::max(this->r, maxer.r);
		this->g = hippt::max(this->g, maxer.g);
		this->b = hippt::max(this->b, maxer.b);
	}

	HIPRT_DEVICE ColorRGB32F maxed(const ColorRGB32F& maxer)
	{
		return ColorRGB32F(hippt::max(this->r, maxer.r), hippt::max(this->g, maxer.g), hippt::max(this->b, maxer.b));
	}

	HIPRT_DEVICE static ColorRGB32F max(const ColorRGB32F& a, const ColorRGB32F& b)
	{
		return ColorRGB32F(hippt::max(a.r, b.r), hippt::max(a.g, b.g), hippt::max(a.b, b.b));
	}

	HIPRT_DEVICE static ColorRGB32F min(const ColorRGB32F& a, const ColorRGB32F& b)
	{
		return ColorRGB32F(hippt::min(a.r, b.r), hippt::min(a.g, b.g), hippt::min(a.b, b.b));
	}

	HIPRT_DEVICE float& operator[](int index)
	{
		return *(&r + index);
	}

	HIPRT_DEVICE float operator[](int index) const
	{
		return *(&r + index);
	}

	HIPRT_DEVICE static ColorRGB32F random_color(unsigned int seed)
	{
		constexpr unsigned int UNSIGNED_INT_MAX = 0xffffffff;

		unsigned int seed1 = wang_hash(seed);
		unsigned int seed2 = wang_hash(seed1);
		unsigned int seed3 = wang_hash(seed2);

		return ColorRGB32F(seed1 / static_cast<float>(UNSIGNED_INT_MAX), seed2 / static_cast<float>(UNSIGNED_INT_MAX),
						   seed3 / static_cast<float>(UNSIGNED_INT_MAX));
	}

	float r, g, b;
};

HIPRT_DEVICE static ColorRGB32F operator+(const ColorRGB32F& a, const ColorRGB32F& b)
{
	return ColorRGB32F(a.r + b.r, a.g + b.g, a.b + b.b);
}

HIPRT_DEVICE static ColorRGB32F operator-(const ColorRGB32F& c)
{
	return ColorRGB32F(-c.r, -c.g, -c.b);
}

HIPRT_DEVICE static ColorRGB32F operator-(const ColorRGB32F& a, const ColorRGB32F& b)
{
	return ColorRGB32F(a.r - b.r, a.g - b.g, a.b - b.b);
}

HIPRT_DEVICE static ColorRGB32F operator*(const ColorRGB32F& a, const ColorRGB32F& b)
{
	return ColorRGB32F(a.r * b.r, a.g * b.g, a.b * b.b);
}

HIPRT_DEVICE static ColorRGB32F operator*(const float k, const ColorRGB32F& c)
{
	return ColorRGB32F(c.r * k, c.g * k, c.b * k);
}

HIPRT_DEVICE static ColorRGB32F operator*(const ColorRGB32F& c, const float k)
{
	return ColorRGB32F(c.r * k, c.g * k, c.b * k);
}

HIPRT_DEVICE static ColorRGB32F operator/(const ColorRGB32F& a, const ColorRGB32F& b)
{
	return ColorRGB32F(a.r / b.r, a.g / b.g, a.b / b.b);
}

HIPRT_DEVICE static ColorRGB32F operator/(const float k, const ColorRGB32F& c)
{
	return ColorRGB32F(k / c.r, k / c.g, k / c.b);
}

HIPRT_DEVICE static ColorRGB32F operator/(const ColorRGB32F& c, const float k)
{
	return ColorRGB32F(c.r / k, c.g / k, c.b / k);
}

HIPRT_DEVICE static ColorRGB32F sqrt(const ColorRGB32F& col)
{
	return ColorRGB32F(hippt::sqrt(col.r), hippt::sqrt(col.g), hippt::sqrt(col.b));
}

HIPRT_DEVICE static ColorRGB32F exp(const ColorRGB32F& col)
{
	return ColorRGB32F(expf(col.r), expf(col.g), expf(col.b));
}

HIPRT_DEVICE static ColorRGB32F exp2(const ColorRGB32F& col)
{
	return ColorRGB32F(exp2f(col.r), exp2f(col.g), exp2f(col.b));
}

HIPRT_DEVICE static ColorRGB32F log(const ColorRGB32F& col)
{
	return ColorRGB32F(logf(col.r), logf(col.g), logf(col.b));
}

HIPRT_DEVICE static ColorRGB32F pow(const ColorRGB32F& col, float k)
{
	return ColorRGB32F(powf(col.r, k), powf(col.g, k), powf(col.b, k));
}

HIPRT_DEVICE static ColorRGB32F intrin_expf(ColorRGB32F x)
{
	return ColorRGB32F(hippt::intrin_expf(x.r), hippt::intrin_expf(x.g), hippt::intrin_expf(x.b));
}

HIPRT_DEVICE static ColorRGB32F intrin_logf(const ColorRGB32F& col)
{
	return ColorRGB32F(hippt::intrin_logf(col.r), hippt::intrin_logf(col.g), hippt::intrin_logf(col.b));
}

HIPRT_DEVICE static ColorRGB32F intrin_pow(ColorRGB32F x, float y)
{
	return ColorRGB32F(hippt::intrin_pow(x.r, y), hippt::intrin_pow(x.g, y), hippt::intrin_pow(x.b, y));
}

HIPRT_DEVICE static ColorRGB32F pow_2_2_fit(ColorRGB32F x)
{
	return ColorRGB32F(hippt::pow_2_2_fit(x.r), hippt::pow_2_2_fit(x.g), hippt::pow_2_2_fit(x.b));
}

#ifndef __KERNELCC__
inline std::ostream& operator<<(std::ostream& os, const ColorRGB32F& color)
{
	os << color.r << ", " << color.g << ", " << color.b;

	return os;
}

inline std::ostream& operator<<(std::ostream& os, const ColorRGBA32F& color)
{
	os << color.r << ", " << color.g << ", " << color.b << ", " << color.a;

	return os;
}
#endif

#endif
