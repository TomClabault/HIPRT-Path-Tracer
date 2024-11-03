/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATH_H
#define HOST_DEVICE_COMMON_MATH_H

#if defined( __KERNELCC__ )
#include <hiprt/hiprt_device.h>
#else
#include <hiprt/hiprt_vec.h>
#endif

#define int2 hiprtInt2
#define int3 hiprtInt3
#define int4 hiprtInt4
#define uint2 hiprtUint2

#define float2 hiprtFloat2
#define float3 hiprtFloat3
#define float4 hiprtFloat4

#define make_int2 make_hiprtInt2
#define make_int3 make_hiprtInt3
#define make_int4 make_hiprtInt4
#define make_uint2 make_hiprtUint2

#define make_float2 make_hiprtFloat2
#define make_float3 make_hiprtFloat3
#define make_float4 make_hiprtFloat4

#if !defined(__KERNELCC__) || defined(HIPRT_BITCODE_LINKING)
#include <hiprt/impl/Math.h>

// For std::atomic in hippt::
#include <atomic>
#endif

struct float4x4
{
	float m[4][4] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };
};

// Here we're defining aliases for common functions used in shader code.
// 
// Because the same shader code can be used both on the CPU and the GPU,
// both code have to compile either through the classical C++ compiler or
// through the GPU shader compiler. This means that we have to use functions
// that were meant to be used on the CPU or on the GPU (depending on the case).
// 
// For example, we're using glm as the math library on the CPU, so 'normalize'
// will actually be aliased to glm::normalize for the CPU
// but 'normalize' will be aliased to hiprt::normalize on the GPU because
// glm isn't meant to be used on the GPU
namespace hippt
{
#ifdef __KERNELCC__
#define M_PI hiprt::Pi
#define M_TWO_PI	6.28318530717958647693f
#define M_INV_PI	0.31830988618379067154f
#define M_INV_2PI	0.63661977236758134308f
#define M_TWO_PIPI	19.73920880217871723767f

#define NEARzero	1.0e-10f

	__device__ bool isZERO(float x) { return (x < NEARzero) && (x > -NEARzero); }
	__device__ float pow5(float x) { return x*x*x*x*x; }

	__device__ float3 cross(float3 u, float3 v) { return hiprt::cross(u, v); }
	__device__ float dot(float3 u, float3 v) { return hiprt::dot(u, v); }

	__device__ float length(float3 u) { return sqrt(hiprt::dot(u, u)); }
	__device__ float length(float2 u) { return sqrt(u.x * u.x + u.y * u.y); }
	__device__ float length2(float3 u) { return hiprt::dot(u, u); }

	__device__ float3 abs(float3 u) { return make_float3(fabsf(u.x), fabsf(u.y), fabsf(u.z)); }
	__device__ float abs(float a) { return fabsf(a); }
	__device__ float max(float a, float b) { return a > b ? a : b; }
	__device__ float min(float a, float b) { return a < b ? a : b; }
	__device__ float clamp(float min_val, float max_val, float val) { return hiprt::clamp(val, min_val, max_val); }

	__device__ float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	__device__ bool isNaN(const T& v) { return isnan(v); }

	template <typename T>
	__device__ T atomic_add(T* address, T increment) { return atomicAdd(address, increment); }

#else
#undef M_PI
#define M_PI		3.14159265358979323846f
#define M_TWO_PI	6.28318530717958647693f
#define M_INV_PI	0.31830988618379067154f
#define M_INV_2PI	0.63661977236758134308f
#define M_TWO_PIPI	19.73920880217871723767f

#define NEARzero	1.0e-10f

	inline bool isZERO(float x) { return (x < NEARzero) && (x > -NEARzero); }
	inline float pow5(float x) { return x*x*x*x*x; }

	inline float3 cross(float3 u, float3 v) { return hiprt::cross(u, v); }
	inline float dot(float3 u, float3 v) { return hiprt::dot(u, v); }

	inline float length(float3 u) { return sqrtf(dot(u, u)); }
	inline float length(float2 u) { return sqrtf(u.x * u.x + u.y * u.y); }
	inline float length2(float3 u) { return dot(u, u); }

	template <typename T>
	inline T max(T a, T b) { return hiprt::max(a, b); }

	template <typename T>
	inline T min(T a, T b) { return hiprt::min(a, b); }

	template <typename T>
	inline T clamp(T min_val, T max_val, T val) { return hiprt::min(max_val, hiprt::max(min_val, val)); }

	inline float3 abs(float3 u) { return make_float3(std::abs(u.x), std::abs(u.y), std::abs(u.z)); }
	inline float abs(float a) { return std::abs(a); }

	inline float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	inline bool isNaN (const T& v) { return std::isnan(v); }

	template <typename T>
	T atomic_add(std::atomic<T>* atomic_address, T increment) { return atomic_address->fetch_add(increment); }
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 matrix_X_point(const float4x4& m, const float3& p)
{
	float x = p.x;
	float y = p.y;
	float z = p.z;

	// Assuming w = 1.0f for the point p
	float xt = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z + m.m[3][0];
	float yt = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z + m.m[3][1];
	float zt = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z + m.m[3][2];
	float wt = m.m[0][3] * x + m.m[1][3] * y + m.m[2][3] * z + m.m[3][3];

	float inv_w = 1.0f;
	if (wt != 0.0f)
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 matrix_X_vec(const float4x4& m, const float3& u)
{
	float x = u.x;
	float y = u.y;
	float z = u.z;

	// Assuming w = 0.0f for the vector u
	float xt = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z;
	float yt = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z;
	float zt = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z;
	float wt = m.m[0][3] * x + m.m[1][3] * y + m.m[2][3] * z;

	float inv_w = 1.0f;
	if (wt != 0.0f)
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

#endif
