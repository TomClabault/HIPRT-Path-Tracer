/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef MATH_H
#define MATH_H

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
#endif

struct float4x4
{
	union
	{
		float4 r[4];
		float  e[4][4];
	};
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
#ifdef __KERNELCC__
namespace hiprtpt
{
	template <typename T>
	__device__ T abs(T val) { return abs(val); }

	template <typename T>
	__device__ T clamp(T min_val, T max_val, T val) { return hiprt::clamp(min_val, max_val, val); }

	template <typename T>
	__device__ T cross(T u, T v) { return hiprt::cross(u, v); }

	template <typename T>
	__device__ float dot(T u, T v) { return hiprt::dot(u, v); }

	template <typename T>
	__device__ float length(T u) { return hiprt::dot(u, u); }

	template <typename T>
	__device__ T max(T a, T b) { return max(a, b); }

	template <typename T>
	__device__ T min(T a, T b) { return min(a, b); }

	template <typename T>
	__device__ T normalize(T u) { return hiprt::normalize(u); }
}

#define M_PI hiprt::Pi
#else
namespace hiprtpt
{
	// TODO use glm instead of gkit
	/*template <typename T>
	T cross(T u, T v) { return glm::cross(u, v); }*/
	template <typename T>
	T cross(T u, T v) { return cross(u, v); }

	// TODO use glm instead of gkit
	/*template <typename T>
	float dot(T u, T v) { return glm::dot(u, v); }*/
	template <typename T>
	float dot(T u, T v) { return dot(u, v); }

	// TODO use glm instead of gkit
	/*template <typename T>
	float length(T u) { return glm::length(u, u); }*/
	template <typename T>
	float length(T u) { return length(u, u); }

	template <typename T>
	T max(T a, T b) { return std::max(a, b); }

	template <typename T>
	T min(T a, T b) { return std::min(a, b); }

	template <typename T>
	T clamp(T min_val, T max_val, T val) { return min(max_val, max(min_val, val)); }

	// TODO use glm instead of gkit
	/*template <typename T>
	T normalize(T u) { return glm::normalize(u); }*/
	template <typename T>
	T normalize(T u) { return normalize(u); }
}
#endif

HIPRT_HOST_DEVICE HIPRT_INLINE float3 matrix_X_point(const float4x4& m, const float3& p)
{
	float x = p.x;
	float y = p.y;
	float z = p.z;

	// Assuming w = 1.0f for the point p
	float xt = m.e[0][0] * x + m.e[1][0] * y + m.e[2][0] * z + m.e[3][0];
	float yt = m.e[0][1] * x + m.e[1][1] * y + m.e[2][1] * z + m.e[3][1];
	float zt = m.e[0][2] * x + m.e[1][2] * y + m.e[2][2] * z + m.e[3][2];
	float wt = m.e[0][3] * x + m.e[1][3] * y + m.e[2][3] * z + m.e[3][3];

	float inv_w = 1.0f;
	if (wt != 0.0f)
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

#endif
