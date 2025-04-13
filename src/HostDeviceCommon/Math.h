/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
// For std::bit_cast in hippt::
#include <bit>
#endif

#ifdef __KERNELCC__
template <typename T>
using AtomicType = T;
#else
#include <atomic>

template <typename T>
using AtomicType = std::atomic<T>;
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
#define M_TWO_PI	6.28318530717958647693f // 2.0f * M_PI
#define M_FOUR_PI	12.5663706143591729539f // 4.0f * M_PI
#define M_INV_PI	0.31830988618379067154f // 1.0f / M_PI
#define M_INV_2_PI	0.15915494309189533577f // 1.0f / (2.0f * M_PI)
#define M_TWO_PI_SQUARED	19.73920880217871723767f
#define NEAR_ZERO	1.0e-10f

	__device__ int thread_idx_x() { return threadIdx.x + blockIdx.x * blockDim.x; }
	__device__ int thread_idx_y() { return threadIdx.y + blockIdx.y * blockDim.y; }
	__device__ bool is_pixel_index(int x, int y) { return hippt::thread_idx_x() == x && hippt::thread_idx_y() == y; }

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

	/**
	 * Component-wise min of float3
	 */
	template <>
	__device__ float3 min(float3 a, float3 b) { return make_float3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }
	/**
	 * Minimum of each component of the float3 against x
	 */
	__device__ float3 min(float3 a, float x) { return make_float3(hiprt::min(a.x, x), hiprt::min(a.y, x), hiprt::min(a.z, x)); }
	__device__ float3 min(float x, float3 a) { return hippt::min(a, x); }

	/**
	 * Component-wise min of int3
	 */
	template <>
	__device__ int3 min(int3 a, int3 b) { return make_int3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }

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
	__device__ float3 ldexp(float3 x, int exp) { return make_float3(ldexpf(x.x, exp), ldexpf(x.y, exp), ldexpf(x.z, exp)); }

	template <typename T>
	__device__ T square(T x) { return x * x; }

	__device__ float2 sqrt(float2 uv) { return make_float2(sqrtf(uv.x), sqrtf(uv.y)); }
	__device__ float3 sqrt(float3 uvw) { return make_float3(sqrtf(uvw.x), sqrtf(uvw.y), sqrtf(uvw.z)); }

	__device__ float pow_1_4(float x) { return sqrtf(sqrtf(x)); }
	__device__ constexpr float pow_3(float x) { return x * x * x; }
	__device__ constexpr float pow_4(float x) { float x2 = x * x; return x2 * x2; }
	__device__ constexpr float pow_5(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
	__device__ constexpr float pow_6(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x2; }

	__device__ float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	__device__ bool is_nan(const T& v) { return isnan(v); }
	template <typename T>
	__device__ bool is_inf(const T& v) { return isinf(v); }
	__device__ bool is_zero(float x) { return x < NEAR_ZERO && x > -NEAR_ZERO; }

	/**
	 * Reads the 32-bit or 64-bit word old located at the address 'address' 
	 * in global or shared memory and stores 'value' to memory at the same address. 
	 * 
	 * These two operations are performed in one atomic transaction. 
	 * The function returns old.
	 */
	template <typename T>
	__device__ T atomic_exchange(T* address, T value) { return atomicExch(address, value); }

	/**
	 * Reads the 32-bit or 64-bit word 'old' located at 'address' in global or shared memory,
	 * computes the maximum of 'old' and 'value', and stores the result back to memory at the
	 * same address.
	 * 
	 * The function returns 'old'
	 */
	template <typename T>
	__device__ T atomic_max(T* address, T value) { return atomicMax(address, value); }

	/**
	 * Reads the 32-bit or 64-bit word 'old' located at 'address' in global or shared memory,
	 * computes the minimum of 'old' and 'value', and stores the result back to memory at the
	 * same address.
	 * 
	 * The function returns 'old'
	 */
	template <typename T> 
	__device__ T atomic_min(T* address, T value) { return atomicMin(address, value); }

	template <typename T>
	__device__ T atomic_fetch_add(T* address, T increment) { return atomicAdd(address, increment); }

	template <typename T>
	__device__ T atomic_load(T* address) { return *address; }
	/**
	 * Reads the 16/32/64 bit word at the 'address' in global or shared memory, 
	 * computes(*address == expected ? new_value : *address), and stores the result
	 * back to memory at the same address. 
	 * 
	 * These three operations are performed in one atomic transaction.
	 * The function returns old (Compare And Swap).
	 */
	template <typename T>
	__device__ T atomic_compare_exchange(T* address, T expected, T new_value) { return atomicCAS(address, expected, new_value); }

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	__device__ T lerp(T a, T b, float t) { return (1.0f - t) * a + t * b; }

	/**
	 * For a 'value' between 'a' and 'b', returns 't' such that
	 * (1.0f - t) * a + t * b = value
	 * 
	 * For 'value' == 'a', returns 0.0f
	 * For 'value' == 'b', returns 1.0f
	 */
	template <typename T>
	__device__ float inverse_lerp(T value, T a, T b) { return (value - a) / (b - a); }

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
	__device__ float asfloat(unsigned int x) { return __uint_as_float(x); }
	__device__ unsigned int asuint(float x) { return __float_as_uint(x); }

	template <typename T>
	__device__ int popc(T bitmask) { return 0; }
	template <>
	__device__ int popc(unsigned int bitmask) { return __popc(bitmask); }
	template <>
	__device__ int popc(unsigned long long int bitmask) { return __popcll(bitmask); }

	/**
	 * Finds the position of least signigicant bit set to 1 in a 32 bit unsigned integer.
	 * Returs a value between 0 and 32 inclusive.
	 *
	 * Returns 0 if all bits are zero
	 */
	__device__ unsigned int ffs(unsigned int bitmask) { return __ffs(bitmask); }

	// TODO these functions require __sync on modern NVIDIA GPUs. We should check that with __CUDACC__
	__device__ bool warp_any(unsigned int thread_mask, bool predicate) { return __any(predicate); }
	__device__ unsigned long long int warp_ballot(unsigned int thread_mask, bool predicate) { return __ballot(predicate); }
	__device__ unsigned int warp_activemask() { return hippt::warp_ballot(0xFFFFFFFF, true); }

	/**
	 * T can be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type.
	 * 
	 * The warp shuffle functions exchange values between threads within a warp.
	 * 
	 * The optional width argument specifies subgroups, in which the warp can be 
	 * divided to share the variables. It has to be a power of two smaller than 
	 * or equal to warpSize. If it is smaller than warpSize, the warp is grouped 
	 * into separate groups, that are each indexed from 0 to width as if it was 
	 * its own entity, and only the lanes within that subgroup participate in the shuffle. 
	 * The lane indices in the subgroup are given by laneIdx % width.
	 * 
	 * 'warp_shfl': The thread reads the value from the lane specified in srcLane
	 */
	template <typename T>
	__device__ T warp_shfl(T var, int src_lane, int width = warpSize) 
	{ 
#ifdef __CUDACC__
		return __shfl_sync(0xFFFFFFFF, var, src_lane, width); 
#else
		return __shfl(var, src_lane, width);
#endif
	}

	/**
	 * Returns the index within its warp (not group) of the calling thread
	 */
	__device__ unsigned int warp_2D_thread_index()
	{
		// warpSize assuming to be a power of 2 so the '&' operation
		// here is a modulo
		return (threadIdx.x + threadIdx.y * blockDim.x) & warpSize;
	}

#else
#undef M_PI
#define M_PI		3.14159265358979323846f
#define M_TWO_PI	6.28318530717958647693f // 2.0f * M_PI
#define M_FOUR_PI	12.5663706143591729539f // 4.0f * M_PI
#define M_INV_PI	0.31830988618379067154f // 1.0f / M_PI
#define M_INV_2_PI	0.15915494309189533577f // 1.0f / (2.0f * M_PI)
#define M_TWO_PI_SQUARED	19.73920880217871723767f // 2.0f * pi^2
#define NEAR_ZERO	1.0e-10f

	inline int thread_idx_x() { return 0; }
	inline int thread_idx_y() { return 0; }
	inline bool is_pixel_index(int x, int y) { return false; }

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

	/**
	 * Component-wise min of float3
	 */
	template <>
	inline float3 min(float3 a, float3 b) { return make_float3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }
	/**
	 * Minimum of each component of the float3 against x
	 */
	inline float3 min(float3 a, float x) { return make_float3(hiprt::min(a.x, x), hiprt::min(a.y, x), hiprt::min(a.z, x)); }
	inline float3 min(float x, float3 a) { return hippt::min(a, x); }

	/**
	 * Component-wise min of int3
	 */
	template <>
	inline int3 min(int3 a, int3 b) { return make_int3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }

	template <typename T>
	inline T clamp(T min_val, T max_val, T val) { return hiprt::min(max_val, hiprt::max(min_val, val)); }

	inline float2 cos(float2 x) { return make_float2(std::cos(x.x), std::cos(x.y)); }
	inline float3 cos(float3 x) { return make_float3(std::cos(x.x), std::cos(x.y), std::cos(x.z)); }

	inline float2 sin(float2 x) { return make_float2(std::sin(x.x), std::sin(x.y)); }
	inline float3 sin(float3 x) { return make_float3(std::sin(x.x), std::sin(x.y), std::sin(x.z)); }

	inline float3 atan2(float3 y, float3 x) { return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z)); }

	inline float2 exp(float2 x) { return make_float2(expf(x.x), expf(x.y)); }
	inline float3 exp(float3 x) { return make_float3(expf(x.x), expf(x.y), expf(x.z)); }
	inline float3 ldexp(float3 x, int exp) { return make_float3(std::ldexp(x.x, exp), std::ldexp(x.y, exp), std::ldexp(x.z, exp)); }

	template <typename T>
	inline T square(T x) { return x * x; }

	inline float2 sqrt(float2 uv) { return make_float2(sqrtf(uv.x), sqrtf(uv.y)); }
	inline float3 sqrt(float3 uvw) { return make_float3(sqrtf(uvw.x), sqrtf(uvw.y), sqrtf(uvw.z)); }
	inline float pow_1_4(float x) { return sqrtf(sqrtf(x)); }
	inline constexpr float pow_3(float x) { return x * x * x; }
	inline constexpr float pow_4(float x) { float x2 = x * x; return x2 * x2; }
	inline constexpr float pow_5(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
	inline constexpr float pow_6(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x2; }

	inline float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	inline bool is_nan(const T& v) { return std::isnan(v); }
	template <typename T>
	inline bool is_inf(const T& v) { return std::isinf(v); }
	inline bool is_zero(float x) { return x < NEAR_ZERO && x > -NEAR_ZERO; }

	/**
	 * Reads the 32-bit or 64-bit word old located at the address 'address'
	 * in global or shared memory and stores 'value' to memory at the same address.
	 *
	 * These two operations are performed in one atomic transaction.
	 * 
	 * The function returns old.
	 */
	template <typename T>
	T atomic_exchange(std::atomic<T>* address, T value) { return address->exchange(value); }

	/**
	 * Reads the 32-bit or 64-bit word 'old' located at 'address' in global or shared memory, 
	 * computes the maximum of 'old' and 'value', and stores the result back to memory at the 
	 * same address. 
	 * 
	 * The function returns 'old'
	 */
	template <typename T>
	T atomic_max(std::atomic<T>* address, T value)
	{
		T prev_value = *address;
		while (prev_value < value && !address->compare_exchange_weak(prev_value, value)) {}

		return prev_value;
	}

	/**
	 * Reads the 32-bit or 64-bit word 'old' located at 'address' in global or shared memory,
	 * computes the minimum of 'old' and 'value', and stores the result back to memory at the
	 * same address.
	 * 
	 * The function returns 'old'
	 */
	template <typename T>
	T atomic_min(std::atomic<T>* address, T value)
	{
		T prev_value = *address;
		while (prev_value > value && !address->compare_exchange_weak(prev_value, value)) {}
		
		return prev_value;
	}

	template <typename T>
	T atomic_fetch_add(std::atomic<T>* atomic_address, T increment) { return atomic_address->fetch_add(increment); }

	template <typename T>
	T atomic_load(std::atomic<T>* atomic_address) { return atomic_address->load(); }

	/**
	 * Reads the 16/32/64 bit word at the 'address' in global or shared memory,
	 * computes(*address == expected ? new_value : *address), and stores the result
	 * back to memory at the same address.
	 *
	 * These three operations are performed in one atomic transaction.
	 * The function returns old (Compare And Swap).
	 */
	template <typename T>
	T atomic_compare_exchange(std::atomic<T>* atomic_address, T expected, T new_value) { return atomic_address->compare_exchange_strong(expected, new_value); }

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	inline T lerp(T a, T b, float t) { return (1.0f - t) * a + t * b; }

	/**
	 * For a 'value' between 'a' and 'b', returns 't' such that
	 * (1.0f - t) * a + t * b = value
	 *
	 * For 'value' == 'a', returns 0.0f
	 * For 'value' == 'b', returns 1.0f
	 */
	template <typename T>
	inline float inverse_lerp(T value, T a, T b) { return (value - a) / (b - a); }
	
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
	inline float asfloat(unsigned int x) { return std::bit_cast<float, unsigned int>(x); }
	inline unsigned int asuint(float x) { return std::bit_cast<unsigned int, float>(x); }
	template <typename T>
	inline int popc(T bitmask) { return std::popcount(bitmask); }

	/**
	 * Finds the position of least signigicant bit set to 1 in a 32 bit unsigned integer.
	 * Returs a value between 0 and 32 inclusive.
	 *
	 * Returns 0 if all bits are zero
	 */
	inline unsigned int ffs(unsigned int bitmask)
	{
		for (int i = 0; i < sizeof(unsigned int) * 8; i++)
			if (bitmask & (1 << i))
				return i;

		return 0;
	}

	inline bool warp_any(unsigned int thread_mask, bool predicate) { return predicate; }
	inline unsigned long long int warp_ballot(unsigned int thread_mask, bool predicate) { return predicate ? 1 : 0; }
	inline unsigned int warp_activemask() { return 1; }

	/**
	 * T can be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type.
	 *
	 * The warp shuffle functions exchange values between threads within a warp.
	 *
	 * The optional width argument specifies subgroups, in which the warp can be
	 * divided to share the variables. It has to be a power of two smaller than
	 * or equal to warpSize. If it is smaller than warpSize, the warp is grouped
	 * into separate groups, that are each indexed from 0 to width as if it was
	 * its own entity, and only the lanes within that subgroup participate in the shuffle.
	 * The lane indices in the subgroup are given by laneIdx % width.
	 *
	 * 'warp_shfl': The thread reads the value from the lane specified in srcLane
	 */
	template <typename T>
	inline T warp_shfl(T var, int srcLane, int width = 1) { return var; }

	/**
	 * Returns the index within its warp (not group) of the calling thread
	 * 
	 * Warp sizes of 1 on the CPU
	 */
	HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int warp_2D_thread_index()
	{
		return 1;
	}
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

#ifndef __KERNELCC__

#include <iostream>
static std::ostream& operator<<(std::ostream& os, float3 uvw)
{
	os << uvw.x << ", " << uvw.y << ", " << uvw.z;
	return os;
}

#endif

#endif
