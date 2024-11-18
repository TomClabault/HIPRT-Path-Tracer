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
	float m[4][4] = { {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f} };
};

struct float3x3
{
	float m[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };
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
#define M_INV_2_PI	0.15915494309189533577f
#define M_TWO_PIPI	19.73920880217871723767f
#define NEAR_ZERO	1.0e-10f

	__device__ float3 cross(float3 u, float3 v) { return hiprt::cross(u, v); }
	__device__ float dot(float3 u, float3 v) { return hiprt::dot(u, v); }

	__device__ float length(float3 u) { return sqrt(hiprt::dot(u, u)); }
	__device__ float length2(float3 u) { return hiprt::dot(u, u); }

	__device__ float3 abs(float3 u) { return make_float3(fabsf(u.x), fabsf(u.y), fabsf(u.z)); }
	__device__ float abs(float a) { return fabsf(a); }

	template <typename T>
	__device__ T max(T a, T b) { return a > b ? a : b; }

	template <>
	__device__ float3 max(float3 a, float3 b) { return make_float3(hippt::max(a.x, b.x), hippt::max(a.y, b.y), hippt::max(a.z, b.z)); }

	template <typename T>
	__device__ T min(T a, T b) { return a < b ? a : b; }

	template <typename T>
	__device__ T clamp(T min_val, T max_val, T val) { return hiprt::min(max_val, hiprt::max(min_val, val)); }

	__device__ float max(float a, float b) { return a > b ? a : b; }
	__device__ float min(float a, float b) { return a < b ? a : b; }
	__device__ float clamp(float min_val, float max_val, float val) { return hiprt::clamp(val, min_val, max_val); }

	__device__ float3 cos(float3 x) { return make_float3(cosf(x.x), cosf(x.y), cosf(x.z)); }
	__device__ float2 cos(float2 x) { return make_float2(cosf(x.x), cosf(x.y)); }

	__device__ float3 sin(float3 x) { return make_float3(sinf(x.x), sinf(x.y), sinf(x.z)); }
	__device__ float2 sin(float2 x) { return make_float2(sinf(x.x), sinf(x.y)); }

	__device__ float3 atan2(float3 y, float3 x) { return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z)); }

	__device__ float2 exp(float2 x) { return make_float2(expf(x.x), expf(x.y)); }
	__device__ float3 exp(float3 x) { return make_float3(expf(x.x), expf(x.y), expf(x.z)); }

	template <typename T>
	__device__ T square(T x) { return x * x; }

	__device__ float2 sqrt(float2 uv) { return make_float2(sqrtf(uv.x), sqrtf(uv.y)); }
	__device__ float3 sqrt(float3 uvw) { return make_float3(sqrtf(uvw.x), sqrtf(uvw.y), sqrtf(uvw.z)); }

	__device__ float pow_1_4(float x) { return sqrtf(sqrtf(x)); }
	__device__ constexpr float pow_4(float x) { float x2 = x * x; return x2 * x2; }
	__device__ constexpr float pow_5(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
	__device__ constexpr float pow_6(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x2; }

	__device__ float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	__device__ bool is_NaN(const T& v) { return isnan(v); }
	__device__ bool is_zero(float x) { return x < NEAR_ZERO && x > -NEAR_ZERO; }

	template <typename T>
	__device__ T atomic_add(T* address, T increment) { return atomicAdd(address, increment); }

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	__device__ T lerp(T a, T b, float t) { return (1.0f - t) * a + t * b; }

	/**
	 * Reference: https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml
	 *
	 * For t == min, returns 0.0f
	 * For t == max, returns 1.0f
	 * Smoothstep interpolation in between
	 */
	template <typename T>
	__device__ T smoothstep(T min, T max, float x)
	{
		float t = hippt::clamp(0.0f, 1.0f, (x - min) / (max - min));

		return t * t * (3.0f - 2.0f * t);
	}

	__device__ float fract(float a) { return a - floorf(a); }

#else
#undef M_PI
#define M_PI		3.14159265358979323846f
#define M_TWO_PI	6.28318530717958647693f
#define M_INV_PI	0.31830988618379067154f
#define M_INV_2_PI	0.15915494309189533577f
#define M_TWO_PIPI	19.73920880217871723767f
#define NEAR_ZERO	1.0e-10f

	inline float3 cross(float3 u, float3 v) { return hiprt::cross(u, v); }
	inline float dot(float3 u, float3 v) { return hiprt::dot(u, v); }

	inline float length(float3 u) { return sqrtf(dot(u, u)); }
	inline float length2(float3 u) { return dot(u, u); }

	inline float3 abs(float3 u) { return make_float3(std::abs(u.x), std::abs(u.y), std::abs(u.z)); }
	inline float abs(float a) { return std::abs(a); }

	template <typename T>
	inline T max(T a, T b) { return hiprt::max(a, b); }

	template <>
	inline float3 max(float3 a, float3 b) { return make_float3(hippt::max(a.x, b.x), hippt::max(a.y, b.y), hippt::max(a.z, b.z)); }

	template <typename T>
	inline T min(T a, T b) { return hiprt::min(a, b); }

	template <typename T>
	inline T clamp(T min_val, T max_val, T val) { return hiprt::min(max_val, hiprt::max(min_val, val)); }

	inline float2 cos(float2 x) { return make_float2(std::cos(x.x), std::cos(x.y)); }
	inline float3 cos(float3 x) { return make_float3(std::cos(x.x), std::cos(x.y), std::cos(x.z)); }

	inline float2 sin(float2 x) { return make_float2(std::sin(x.x), std::sin(x.y)); }
	inline float3 sin(float3 x) { return make_float3(std::sin(x.x), std::sin(x.y), std::sin(x.z)); }

	inline float3 atan2(float3 y, float3 x) { return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z)); }

	inline float2 exp(float2 x) { return make_float2(expf(x.x), expf(x.y)); }
	inline float3 exp(float3 x) { return make_float3(expf(x.x), expf(x.y), expf(x.z)); }

	template <typename T>
	inline T square(T x) { return x * x; }

	inline float2 sqrt(float2 uv) { return make_float2(sqrtf(uv.x), sqrtf(uv.y)); }
	inline float3 sqrt(float3 uvw) { return make_float3(sqrtf(uvw.x), sqrtf(uvw.y), sqrtf(uvw.z)); }
	inline float pow_1_4(float x) { return sqrtf(sqrtf(x)); }
	inline constexpr float pow_4(float x) { float x2 = x * x; return x2 * x2; }
	inline constexpr float pow_5(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
	inline constexpr float pow_6(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x2; }

	inline float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	inline bool is_NaN (const T& v) { return std::isnan(v); }
	inline bool is_zero(float x) { return x < NEAR_ZERO && x > -NEAR_ZERO; }

	template <typename T>
	T atomic_add(std::atomic<T>* atomic_address, T increment) { return atomic_address->fetch_add(increment); }

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	inline T lerp(T a, T b, float t) { return (1.0f - t) * a + t * b; }
	
	/**
	 * Reference: https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml
	 * 
	 * For t == min, returns 0.0f
	 * For t == max, returns 1.0f
	 * Smoothstep interpolation in between
	 */
	template <typename T>
	inline T smoothstep(T min, T max, float x) 
	{ 
		float t = hippt::clamp(0.0f, 1.0f, (x - min) / (max - min));

		return t * t * (3.0f - 2.0f * t);
	}

	inline float fract(float a) { return a - floorf(a); }
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 matrix_X_point(const float4x4& m, const float3& p)
{
	float x = p.x;
	float y = p.y;
	float z = p.z;

	// Assuming w = 1.0f for the point p
	float xt = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
	float yt = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
	float zt = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
	float wt = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];

	float inv_w = 1.0f;
	if (!hippt::is_zero(wt))
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 matrix_X_vec(const float3x3& m, const float3& u)
{
	float x = u.x;
	float y = u.y;
	float z = u.z;

	// Assuming w = 0.0f for the vector u
	float xt = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z;
	float yt = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z;
	float zt = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z;

	return make_float3(xt, yt, zt);
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
	if (!hippt::is_zero(wt))
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

#endif
