/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_VEC_TYPES_H
#define HOST_DEVICE_COMMON_VEC_TYPES_H

#ifndef __KERNELCC__

#include <hiprt/hiprt_vec.h>

template <typename T, uint32_t N>
struct math_vector
{
};

template <typename T>
struct math_vector<T, 2>
{
	math_vector() = default;
	constexpr math_vector<T, 2>(T v) : x(v), y(v) {}
	constexpr math_vector<T, 2>(T x_, T y_) : x(x_), y(y_) {}
	constexpr math_vector<T, 2>(const hiprt::Vector<T, 2>& v) : x(v.x), y(v.y) {}

	operator hiprt::Vector<T, 2>() const
	{
		return hiprt::Vector<T, 2>{ x, y };
	}

	math_vector<T, 2>& operator+=(const math_vector<T, 2>& other)
	{
		x += other.x;
		y += other.y;

		return *this;
	}

	math_vector<T, 2>& operator-=(const math_vector<T, 2>& other)
	{
		x -= other.x;
		y -= other.y;

		return *this;
	}

	math_vector<T, 2>& operator*=(const T& k)
	{
		x *= k;
		y *= k;

		return *this;
	}

	math_vector<T, 2>& operator*=(const math_vector<T, 2>& other)
	{
		x *= other.x;
		y *= other.y;

		return *this;
	}

	math_vector<T, 2>& operator/=(const T& k)
	{
		x /= k;
		y /= k;

		return *this;
	}

	T x, y;
};

template <typename T>
struct math_vector<T, 3>
{
	math_vector() = default;
	constexpr math_vector<T, 3>(T v) : x(v), y(v), z(v) {}
	constexpr math_vector<T, 3>(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
	constexpr math_vector<T, 3>(const hiprt::Vector<T, 3>& v) : x(v.x), y(v.y), z(v.z) {}

	operator hiprt::Vector<T, 3>() const
	{
		return hiprt::Vector<T, 3>{ x, y, z };
	}

	math_vector<T, 3>& operator+=(const math_vector<T, 3>& other)
	{
		x += other.x;
		y += other.y;
		z += other.z;

		return *this;
	}

	math_vector<T, 3>& operator-=(const math_vector<T, 3>& other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;

		return *this;
	}

	math_vector<T, 3>& operator*=(const T& k)
	{
		x *= k;
		y *= k;
		z *= k;

		return *this;
	}

	math_vector<T, 3>& operator*=(const math_vector<T, 3>& other)
	{
		x *= other.x;
		y *= other.y;
		z *= other.z;

		return *this;
	}

	math_vector<T, 3>& operator/=(const T& k)
	{
		x /= k;
		y /= k;
		z /= k;

		return *this;
	}

	T x, y, z;
};

template <typename T>
struct math_vector<T, 4>
{
	math_vector() = default;
	constexpr math_vector<T, 4>(T v) : x(v), y(v), z(v), w(v) {}
	constexpr math_vector<T, 4>(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
	constexpr math_vector<T, 4>(const hiprt::Vector<T, 4>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

	operator hiprt::Vector<T, 4>() const
	{
		return hiprt::Vector<T, 2>{ x, y, z, w };
	}

	math_vector<T, 4>& operator+=(const math_vector<T, 4>& other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
		w += other.w;

		return *this;
	}

	math_vector<T, 4>& operator-=(const math_vector<T, 4>& other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
		w -= other.w;

		return *this;
	}

	math_vector<T, 4>& operator*=(const T& k)
	{
		x *= k;
		y *= k;
		z *= k;
		w *= k;

		return *this;
	}

	math_vector<T, 4>& operator*=(const math_vector<T, 4>& other)
	{
		x *= other.x;
		y *= other.y;
		z *= other.z;
		w *= other.w;

		return *this;
	}

	math_vector<T, 4>& operator/=(const T& k)
	{
		x /= k;
		y /= k;
		z /= k;
		w /= k;

		return *this;
	}

	T x, y, z, w;
};

template <typename T>
math_vector<T, 2> operator+(const math_vector<T, 2>& a, const math_vector<T, 2>& b)
{
	return math_vector<T, 2>{ a.x + b.x, a.y + b.y };
}

template <typename T>
math_vector<T, 2> operator+(const math_vector<T, 2>& a, const hiprt::Vector<T, 2>& b)
{
	return math_vector<T, 2>{ a.x + b.x, a.y + b.y };
}

template <typename T>
math_vector<T, 2> operator+(const math_vector<T, 2>& vec, T b)
{
	return math_vector<T, 2>{ vec.x + b, vec.y + b };
}

template <typename T>
hiprt::Vector<T, 2> operator+(const hiprt::Vector<T, 2>& a, const hiprt::Vector<T, 2>& b)
{
	return hiprt::Vector<T, 2>{ a.x + b.x, a.y + b.y };
}

template <typename T>
math_vector<T, 2> operator-(const math_vector<T, 2>& a, const math_vector<T, 2>& b)
{
	return math_vector<T, 2>{ a.x - b.x, a.y - b.y };
}

template <typename T>
math_vector<T, 2> operator-(const math_vector<T, 2>& a, const hiprt::Vector<T, 2>& vec)
{
	return math_vector<T, 2>{ a.x - vec.x, a.y - vec.y };
}

template <typename T>
math_vector<T, 2> operator-(const hiprt::Vector<T, 2>& vec, const math_vector<T, 2>& a)
{
	return math_vector<T, 2>{ vec.x - a.x, vec.y - a.y };
}

template <typename T>
math_vector<T, 2> operator-(const math_vector<T, 2>& vec, T a)
{
	return math_vector<T, 2>{ vec.x - a, vec.y - a };
}

template <typename T>
math_vector<T, 2> operator-(const math_vector<T, 2>& a)
{
	return math_vector<T, 2>{ -a.x, -a.y };
}

template <typename T>
math_vector<T, 2> operator-(const hiprt::Vector<T, 2>& a)
{
	return math_vector<T, 2>{ -a.x, -a.y };
}

template <typename T>
math_vector<T, 2> operator*(const math_vector<T, 2>& a, T val)
{
	return math_vector<T, 2>{ a.x * val, a.y * val };
}

template <typename T>
math_vector<T, 2> operator*(T val, const math_vector<T, 2>& a)
{
	return a * val;
}

template <typename T>
math_vector<T, 2> operator*(const hiprt::Vector<T, 2>& a, T b)
{
	return math_vector<T, 2>{ a.x * b, a.y * b };
}

template <typename T>
math_vector<T, 2> operator*(const T& a, const hiprt::Vector<T, 2>& b)
{
	return b * a;
}

template <typename T>
math_vector<T, 2> operator*(const math_vector<T, 2>& a, const math_vector<T, 2>& b)
{
	return math_vector<T, 2>{ a.x * b.x, a.y * b.y };
}

template <typename T>
math_vector<T, 2> operator/(const math_vector<T, 2>& a, T b)
{
	return math_vector<T, 2>{ a.x / b, a.y / b };
}

/*
 * Size 3 begins
 */

template <typename T>
math_vector<T, 3> operator+(const math_vector<T, 3>& a, const math_vector<T, 3>& b)
{
	return math_vector<T, 3>{ a.x + b.x, a.y + b.y, a.z + b.z };
}

template <typename T>
math_vector<T, 3> operator+(const math_vector<T, 3>& a, hiprt::Vector<T, 3> vec)
{
	return math_vector<T, 3>{ a.x + vec.x, a.y + vec.y, a.z + vec.z };
}

template <typename T>
math_vector<T, 3> operator+(const hiprt::Vector<T, 3>& vec, const math_vector<T, 3>& a)
{
	return a + vec;
}

template <typename T>
math_vector<T, 3> operator+(const math_vector<T, 3>& vec, T b)
{
	return math_vector<T, 3>{ vec.x + b, vec.y + b, vec.z + b };
}

template <typename T>
hiprt::Vector<T, 3> operator+(const hiprt::Vector<T, 3>& a, const hiprt::Vector<T, 3>& b)
{
	return hiprt::Vector<T, 3>{ a.x + b.x, a.y + b.y, a.z + b.z };
}

template <typename T>
math_vector<T, 3> operator-(const math_vector<T, 3>& a, const math_vector<T, 3>& b)
{
	return math_vector<T, 3>{ a.x - b.x, a.y - b.y, a.z - b.z };
}

template <typename T>
math_vector<T, 3> operator-(const math_vector<T, 3>& a, const hiprt::Vector<T, 3>& vec)
{
	return math_vector<T, 3>{ a.x - vec.x, a.y - vec.y, a.z - vec.z };
}

template <typename T>
math_vector<T, 3> operator-(const hiprt::Vector<T, 3>& vec, const math_vector<T, 3>& a)
{
	return math_vector<T, 3>{ vec.x - a.x, vec.y - a.y, vec.z - a.z };
}

template <typename T>
math_vector<T, 3> operator-(const math_vector<T, 3>& vec, T a)
{
	return math_vector<T, 3>{ vec.x - a, vec.y - a, vec.z - a };
}

template <typename T>
math_vector<T, 3> operator-(const math_vector<T, 3>& a)
{
	return math_vector<T, 3>{ -a.x, -a.y, -a.z };
}

template <typename T>
math_vector<T, 3> operator-(const hiprt::Vector<T, 3>& a)
{
	return math_vector<T, 3>{ -a.x, -a.y, -a.z };
}

template <typename T>
math_vector<T, 3> operator*(const math_vector<T, 3>& a, T val)
{
	return math_vector<T, 3>{ a.x * val, a.y * val, a.z * val };
}

template <typename T>
math_vector<T, 3> operator*(T val, const math_vector<T, 3>& a)
{
	return a * val;
}

template <typename T>
math_vector<T, 3> operator*(const hiprt::Vector<T, 3>& a, T b)
{
	return math_vector<T, 3>{ a.x * b, a.y * b, a.z * b };
}

template <typename T>
math_vector<T, 3> operator*(const T& a, const hiprt::Vector<T, 3>& b)
{
	return b * a;
}

template <typename T>
math_vector<T, 3> operator*(const math_vector<T, 3>& a, const math_vector<T, 3>& b)
{
	return math_vector<T, 3>{ a.x * b.x, a.y * b.y, a.z * b.z };
}

template <typename T>
math_vector<T, 3> operator/(const math_vector<T, 3>& a, T b)
{
	return math_vector<T, 3>{ a.x / b, a.y / b, a.z / b };
}

/*
 * Size 4 begins
 */

template <typename T>
math_vector<T, 4> operator+(const math_vector<T, 4>& a, const math_vector<T, 4>& b)
{
	return math_vector<T, 4>{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

template <typename T>
math_vector<T, 4> operator+(const math_vector<T, 4>& a, hiprt::Vector<T, 4> vec)
{
	return math_vector<T, 4>{ a.x + vec.x, a.y + vec.y, a.z + vec.z, a.w + vec.w };
}

template <typename T>
math_vector<T, 4> operator+(const hiprt::Vector<T, 4>& vec, const math_vector<T, 4>& a)
{
	return a + vec;
}

template <typename T>
math_vector<T, 4> operator+(const math_vector<T, 4>& vec, T b)
{
	return math_vector<T, 4>{ vec.x + b, vec.y + b, vec.z + b, vec.w + b };
}

template <typename T>
hiprt::Vector<T, 4> operator+(const hiprt::Vector<T, 4>& a, const hiprt::Vector<T, 4>& b)
{
	return hiprt::Vector<T, 4>{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

template <typename T>
math_vector<T, 4> operator-(const math_vector<T, 4>& a, const math_vector<T, 4>& b)
{
	return math_vector<T, 4>{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

template <typename T>
math_vector<T, 4> operator-(const math_vector<T, 4>& a, const hiprt::Vector<T, 4>& vec)
{
	return math_vector<T, 4>{ a.x - vec.x, a.y - vec.y, a.z - vec.z, a.w - vec.w };
}

template <typename T>
math_vector<T, 4> operator-(const hiprt::Vector<T, 4>& vec, const math_vector<T, 4>& a)
{
	return math_vector<T, 4>{ vec.x - a.x, vec.y - a.y, vec.z - a.z, vec.w - a.w };
}

template <typename T>
math_vector<T, 4> operator-(const math_vector<T, 4>& vec, T a)
{
	return math_vector<T, 4>{ vec.x - a, vec.y - a, vec.z - a, vec.w - a };
}

template <typename T>
math_vector<T, 4> operator-(const math_vector<T, 4>& a)
{
	return math_vector<T, 4>{ -a.x, -a.y, -a.z, -a.w };
}

template <typename T>
math_vector<T, 4> operator-(const hiprt::Vector<T, 4>& a)
{
	return math_vector<T, 4>{ -a.x, -a.y, -a.z, -a.w };
}

template <typename T>
math_vector<T, 4> operator*(const math_vector<T, 4>& a, T b)
{
	return math_vector<T, 4>{ a.x * b, a.y * b, a.z * b, a.w * b };
}

template <typename T>
math_vector<T, 4> operator*(T b, const math_vector<T, 4>& a)
{
	return b * a;
}

template <typename T>
math_vector<T, 4> operator*(const hiprt::Vector<T, 4>& a, T b)
{
	return math_vector<T, 4>{ a.x * b, a.y * b, a.z * b, a.w * b };
}

template <typename T>
math_vector<T, 4> operator*(T a, const hiprt::Vector<T, 4>& b)
{
	return b * a;
}

template <typename T>
math_vector<T, 4> operator*(const math_vector<T, 4>& a, const math_vector<T, 4>& b)
{
	return math_vector<T, 4>{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

template <typename T>
math_vector<T, 4> operator/(const math_vector<T, 4>& a, T b)
{
	return math_vector<T, 4>{ a.x / b, a.y / b, a.z / b, a.w / b };
}

using int2	 = math_vector<int, 2>;
using int3	 = math_vector<int, 3>;
using int4	 = math_vector<int, 4>;
using uint2	 = math_vector<unsigned int, 2>;
using uint3	 = math_vector<unsigned int, 3>;
using uint4	 = math_vector<unsigned int, 4>;
using float2 = math_vector<float, 2>;
using float3 = math_vector<float, 3>;
using float4 = math_vector<float, 4>;

inline constexpr int2 make_int2(int x, int y)
{
	return int2(x, y);
}
inline constexpr int2 make_int2(int v)
{
	return int2(v, v);
}
inline constexpr int3 make_int3(int x, int y, int z)
{
	return int3(x, y, z);
}
inline constexpr int3 make_int3(int v)
{
	return int3(v, v, v);
}
inline constexpr int4 make_int4(int x, int y, int z, int w)
{
	return int4(x, y, z, w);
}
inline constexpr int4 make_int4(int v)
{
	return int4(v, v, v, v);
}

inline constexpr uint2 make_uint2(unsigned int x, unsigned int y)
{
	return uint2(x, y);
}
inline constexpr uint2 make_uint2(unsigned int v)
{
	return uint2(v, v);
}
inline constexpr uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
	return uint3(x, y, z);
};
inline constexpr uint3 make_uint3(unsigned int v)
{
	return uint3(v, v, v);
}
inline constexpr uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
	return uint4(x, y, z, w);
}
inline constexpr uint4 make_uint4(unsigned int v)
{
	return uint4(v, v, v, v);
}

inline constexpr float2 make_float2(float x, float y)
{
	return float2(x, y);
}
inline constexpr float2 make_float2(float v)
{
	return float2(v, v);
}
inline constexpr float3 make_float3(float x, float y, float z)
{
	return float3(x, y, z);
}
inline constexpr float3 make_float3(float v)
{
	return float3(v, v, v);
}
inline constexpr float4 make_float4(float x, float y, float z, float w)
{
	return float4(x, y, z, w);
}
inline constexpr float4 make_float4(float v)
{
	return float4(v, v, v, v);
}

#else // __KERNELCC__

#include "Device/includes/FixIntellisense.h"

// Defining the missing one-value constructors
HIPRT_DEVICE inline constexpr int2 make_int2(int v)
{
	return make_int2(v, v);
}
HIPRT_DEVICE inline constexpr int3 make_int3(int v)
{
	return make_int3(v, v, v);
}
HIPRT_DEVICE inline constexpr int4 make_int4(int v)
{
	return make_int4(v, v, v, v);
}

HIPRT_DEVICE inline constexpr uint2 make_uint2(unsigned int v)
{
	return make_uint2(v, v);
}
HIPRT_DEVICE inline constexpr uint3 make_uint3(unsigned int v)
{
	return make_uint3(v, v, v);
}
HIPRT_DEVICE inline constexpr uint4 make_uint4(unsigned int v)
{
	return make_uint4(v, v, v, v);
}

HIPRT_DEVICE inline constexpr float2 make_float2(float v)
{
	return make_float2(v, v);
}
HIPRT_DEVICE inline constexpr float3 make_float3(float v)
{
	return make_float3(v, v, v);
}
HIPRT_DEVICE inline constexpr float4 make_float4(float v)
{
	return make_float4(v, v, v, v);
}

#endif // !__KERNELCC__

#endif
