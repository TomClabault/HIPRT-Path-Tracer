/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
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
	constexpr explicit math_vector<T, 2>(T v) : x(v), y(v) {}
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
	constexpr explicit math_vector<T, 3>(T v) : x(v), y(v), z(v) {}
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
	constexpr explicit math_vector<T, 4>(T v) : x(v), y(v), z(v), w(v) {}
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

inline math_vector<float, 3> operator+(const hiprtFloat3& a, const math_vector<float, 3>& b)
{
	return math_vector<float, 3>{ a.x + b.x, a.y + b.y, a.z + b.z };
}

inline math_vector<float, 3> operator+(const math_vector<float, 3>& a, const hiprtFloat3& b)
{
	return b + a;
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

inline math_vector<float, 3> operator-(const hiprtFloat3& a, const math_vector<float, 3>& b)
{
	return math_vector<float, 3>{ a.x - b.x, a.y - b.y, a.z - b.z };
}

inline math_vector<float, 3> operator-(const hiprtFloat3& a, const hiprtFloat3& b)
{
	return math_vector<float, 3>{ a.x - b.x, a.y - b.y, a.z - b.z };
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

inline math_vector<float, 3> operator*(const hiprtFloat3& a, float b)
{
	return math_vector<float, 3>{ a.x * b, a.y * b, a.z * b };
}

inline math_vector<float, 3> operator*(float a, const hiprtFloat3& b)
{
	return b * a;
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

using uchar2_t	= math_vector<unsigned char, 2>;
using uchar3_t	= math_vector<unsigned char, 3>;
using uchar4_t	= math_vector<unsigned char, 4>;
using short2_t	= math_vector<short int, 2>;
using short3_t	= math_vector<short int, 3>;
using short4_t	= math_vector<short int, 4>;
using int2_t	= math_vector<int, 2>;
using int3_t	= math_vector<int, 3>;
using int4_t	= math_vector<int, 4>;
using uint2_t	= math_vector<unsigned int, 2>;
using uint3_t	= math_vector<unsigned int, 3>;
using uint4_t	= math_vector<unsigned int, 4>;
using float2_t	= math_vector<float, 2>;
using float3_t	= math_vector<float, 3>;
using float4_t	= math_vector<float, 4>;
using double2_t = math_vector<double, 2>;
using double3_t = math_vector<double, 3>;
using double4_t = math_vector<double, 4>;

inline constexpr uchar2_t make_uchar2(unsigned char x, unsigned char y)
{
	return uchar2_t(x, y);
}

inline constexpr uchar2_t make_uchar2(unsigned char v)
{
	return uchar2_t(v, v);
}

inline constexpr uchar3_t make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
	return uchar3_t(x, y, z);
}

inline constexpr uchar3_t make_uchar3(unsigned char v)
{
	return uchar3_t(v, v, v);
}

inline constexpr uchar4_t make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
	return uchar4_t(x, y, z, w);
}

inline constexpr uchar4_t make_uchar4(unsigned char v)
{
	return uchar4_t(v, v, v, v);
}

inline constexpr short2_t make_short2(short int x, short int y)
{
	return short2_t(x, y);
}

inline constexpr short2_t make_short2(short int v)
{
	return short2_t(v, v);
}

inline constexpr short3_t make_short3(short int x, short int y, short int z)
{
	return short3_t(x, y, z);
}

inline constexpr short3_t make_short3(short int v)
{
	return short3_t(v, v, v);
}

inline constexpr short4_t make_short4(short int x, short int y, short int z, short int w)
{
	return short4_t(x, y, z, w);
}

inline constexpr short4_t make_short4(short int v)
{
	return short4_t(v, v, v, v);
}

inline constexpr int2_t make_int2(int x, int y)
{
	return int2_t(x, y);
}

inline constexpr int2_t make_int2(int v)
{
	return int2_t(v, v);
}

inline constexpr int3_t make_int3(int x, int y, int z)
{
	return int3_t(x, y, z);
}

inline constexpr int3_t make_int3(int v)
{
	return int3_t(v, v, v);
}

inline constexpr int4_t make_int4(int x, int y, int z, int w)
{
	return int4_t(x, y, z, w);
}

inline constexpr int4_t make_int4(int v)
{
	return int4_t(v, v, v, v);
}

inline constexpr uint2_t make_uint2(unsigned int x, unsigned int y)
{
	return uint2_t(x, y);
}

inline constexpr uint2_t make_uint2(unsigned int v)
{
	return uint2_t(v, v);
}

inline constexpr uint3_t make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
	return uint3_t(x, y, z);
};
inline constexpr uint3_t make_uint3(unsigned int v)
{
	return uint3_t(v, v, v);
}

inline constexpr uint4_t make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
	return uint4_t(x, y, z, w);
}

inline constexpr uint4_t make_uint4(unsigned int v)
{
	return uint4_t(v, v, v, v);
}

inline constexpr float2_t make_float2(float x, float y)
{
	return float2_t(x, y);
}

inline constexpr float2_t make_float2(float v)
{
	return float2_t(v, v);
}

inline constexpr float3_t make_float3(float x, float y, float z)
{
	return float3_t(x, y, z);
}

inline constexpr float3_t make_float3(float v)
{
	return float3_t(v, v, v);
}

inline constexpr float4_t make_float4(float x, float y, float z, float w)
{
	return float4_t(x, y, z, w);
}

inline constexpr float4_t make_float4(float v)
{
	return float4_t(v, v, v, v);
}

inline constexpr double2_t make_double2(double x, double y)
{
	return double2_t(x, y);
}

inline constexpr double2_t make_double2(double v)
{
	return double2_t(v, v);
}

inline constexpr double3_t make_double3(double x, double y, double z)
{
	return double3_t(x, y, z);
}

inline constexpr double3_t make_double3(double v)
{
	return double3_t(v, v, v);
}

inline constexpr double4_t make_double4(double x, double y, double z, double w)
{
	return double4_t(x, y, z, w);
}

inline constexpr double4_t make_double4(double v)
{
	return double4_t(v, v, v, v);
}

#else // __KERNELCC__

#include "Device/includes/FixIntellisense.h"

using uchar2_t	= uchar2;
using uchar3_t	= uchar3;
using uchar4_t	= uchar4;
using short2_t	= short2;
using short3_t	= short3;
using short4_t	= short4;
using int2_t	= int2;
using int3_t	= int3;
using int4_t	= int4;
using uint2_t	= uint2;
using uint3_t	= uint3;
using uint4_t	= uint4;
using float2_t	= float2;
using float3_t	= float3;
using float4_t	= float4;
using double2_t = double2;
using double3_t = double3;
using double4_t = double4;

// Defining the missing one-value constructors
HIPRT_DEVICE inline constexpr uchar2_t make_uchar2(unsigned char v)
{
	return uchar2_t(v, v);
}

HIPRT_DEVICE inline constexpr uchar3_t make_uchar3(unsigned char v)
{
	return uchar3_t(v, v, v);
}

HIPRT_DEVICE inline constexpr uchar4_t make_uchar4(unsigned char v)
{
	return uchar4_t(v, v, v, v);
}

HIPRT_DEVICE inline constexpr short2_t make_short2(short int v)
{
	return make_short2(v, v);
}

HIPRT_DEVICE inline constexpr short3_t make_short3(short int v)
{
	return make_short3(v, v, v);
}

HIPRT_DEVICE inline constexpr short4_t make_short4(short int v)
{
	return make_short4(v, v, v, v);
}

HIPRT_DEVICE inline constexpr int2_t make_int2(int v)
{
	return make_int2(v, v);
}

HIPRT_DEVICE inline constexpr int3_t make_int3(int v)
{
	return make_int3(v, v, v);
}

HIPRT_DEVICE inline constexpr int4_t make_int4(int v)
{
	return make_int4(v, v, v, v);
}

HIPRT_DEVICE inline constexpr uint2_t make_uint2(unsigned int v)
{
	return make_uint2(v, v);
}

HIPRT_DEVICE inline constexpr uint3_t make_uint3(unsigned int v)
{
	return make_uint3(v, v, v);
}

HIPRT_DEVICE inline constexpr uint4_t make_uint4(unsigned int v)
{
	return make_uint4(v, v, v, v);
}

HIPRT_DEVICE inline constexpr float2_t make_float2(float v)
{
	return make_float2(v, v);
}

HIPRT_DEVICE inline constexpr float3_t make_float3(float v)
{
	return make_float3(v, v, v);
}

HIPRT_DEVICE inline constexpr float4_t make_float4(float v)
{
	return make_float4(v, v, v, v);
}

HIPRT_DEVICE inline constexpr double2_t make_double2(double v)
{
	return make_double2(v, v);
}

HIPRT_DEVICE inline constexpr double3_t make_double3(double v)
{
	return make_double3(v, v, v);
}

HIPRT_DEVICE inline constexpr double4_t make_double4(double v)
{
	return make_double4(v, v, v, v);
}

#endif // !__KERNELCC__

#endif
