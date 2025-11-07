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

// For hippt::debugbreak()
#include "Utils/Debug.h"
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

#include "HostDeviceCommon/AtomicType.h"

struct float4x4
{
	float m[4][4] = { {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f} };
};

struct float3x3
{
	float m[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };
};

struct float2x2
{
	HIPRT_DEVICE float2x2() {}
	HIPRT_DEVICE float2x2(float m00, float m01, float m10, float m11)
	{
		m[0][0] = m00; m[0][1] = m01;
		m[1][0] = m10; m[1][1] = m11;
	}

	/**
	 * Construct from 2 rows
	 */
	HIPRT_DEVICE float2x2(float2 col0, float2 col1)
	{
		m[0][0] = col0.x; m[0][1] = col1.x;
		m[1][0] = col0.y; m[1][1] = col1.y;
	}

	float m[2][2];
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
	constexpr float M_INV_TWO_PI		= 0.15915494309189533577f;	// 1.0f / (2.0f * M_PI)
	constexpr float M_INV_PI			= 0.31830988618379067154f;	// 1.0f / M_PI
	constexpr float M_PI_TWO			= 1.57079632679489661923;	// pi/2
	constexpr float M_Pi				= 3.14159265358979323846;	// pi
	constexpr float M_TWO_PI			= 6.28318530717958647693f;	// 2.0f * M_PI
	constexpr float M_FOUR_PI			= 12.5663706143591729539f;	// 4.0f * M_PI
	constexpr float M_TWO_PI_SQUARED	= 19.73920880217871723767f;	// 2.0f * M_PI ^ 2
	constexpr float NEAR_ZERO			= 1.0e-10f;

	constexpr float FLOAT_MAX = 3.402823466e+38f;
	constexpr float FLOAT_MIN = 1.175494351e-38f;
	constexpr float FLOAT_EPSILON = 1.192092896e-07f;

	__device__ float Infinity() { return __int_as_float(0x7f800000); }

	/**
	 * Returns the 'warpSize' runtime constant of the GPU
	 */
	__device__ int warp_size() { return warpSize; }
	__device__ int thread_idx_x() { return threadIdx.x + blockIdx.x * blockDim.x; }
	__device__ int thread_idx_y() { return threadIdx.y + blockIdx.y * blockDim.y; }
	__device__ int thread_idx_global() { return hippt::thread_idx_x() + hippt::thread_idx_y() * blockDim.x * gridDim.x; }
	// __device__ bool is_pixel_index(int x, int y) { return hippt::thread_idx_x() == x && hippt::thread_idx_y() == y; }
	__device__ bool is_pixel_index(int x, int y) { return false; }
	__device__ int current_warp_lane() { return (threadIdx.x + threadIdx.y * blockDim.x) % hippt::warp_size(); }

	template <typename T>
	__device__ T ldg_load(T* address) { return __ldg(address); }

	__device__ float3 cross(float3 u, float3 v) { return hiprt::cross(u, v); }
	__device__ float dot(float3 u, float3 v) { return hiprt::dot(u, v); }
	__device__ float dot(float2 u, float2 v) { return u.x * v.x + u.y * v.y; }

	__device__ float length(float3 u) { return sqrt(hiprt::dot(u, u)); }
	__device__ float length2(float3 u) { return hiprt::dot(u, u); }

	__device__ float3 abs(float3 u) { return make_float3(fabsf(u.x), fabsf(u.y), fabsf(u.z)); }
	__device__ float abs(float a) { return fabsf(a); }

	/**
	 * a * b + c
	 */
	__device__ float fma(float a, float b, float c) { return fmaf(a, b, c); }
	__device__ float2 fma(float2 a, float2 b, float2 c) { return make_float2(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y)); }
	__device__ float3 fma(float3 a, float3 b, float3 c) { return make_float3(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y), hippt::fma(a.z, b.z, c.z)); }



	template <typename T>
	__device__ T max(T a, T b) { return a > b ? a : b; }

	/**
	 * Component-wise max of float3 and int3
	 */
	template <>
	__device__ float3 max(float3 a, float3 b) { return make_float3(hiprt::max(a.x, b.x), hiprt::max(a.y, b.y), hiprt::max(a.z, b.z)); }
	template <>
	__device__ int3 max(int3 a, int3 b) { return make_int3(hiprt::max(a.x, b.x), hiprt::max(a.y, b.y), hiprt::max(a.z, b.z)); }





	template <typename T>
	__device__ T min(T a, T b) { return a < b ? a : b; }

	/**
	 * Component-wise min of float3 and int3
	 */
	template <>
	__device__ float3 min(float3 a, float3 b) { return make_float3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }
	template <>
	__device__ int3 min(int3 a, int3 b) { return make_int3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }
	/**
	 * Minimum of each component of the float3 against x
	 */
	__device__ float3 min(float3 a, float x) { return make_float3(hiprt::min(a.x, x), hiprt::min(a.y, x), hiprt::min(a.z, x)); }
	__device__ float3 min(float x, float3 a) { return hippt::min(a, x); }

	/**
	 * Minimum of each component of the mat 2x2 against x
	 */
	__device__ float2x2 min(float x, float2x2 a) { return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x)); }
	__device__ float2x2 min(float2x2 a, float x) { return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x)); }

	template <typename T>
	__device__ T clamp(T min_val, T max_val, T val) { return hiprt::min(max_val, hiprt::max(min_val, val)); }

	__device__ float max(float a, float b) { return a > b ? a : b; }
	__device__ float min(float a, float b) { return a < b ? a : b; }





	__device__ float clamp(float min_val, float max_val, float val) { return hiprt::clamp(val, min_val, max_val); }

	__device__ float3 cos(float3 x) { return make_float3(cosf(x.x), cosf(x.y), cosf(x.z)); }
	__device__ float2 cos(float2 x) { return make_float2(cosf(x.x), cosf(x.y)); }
	__device__ float intrin_cosf(float x) { return __cosf(x); }
	__device__ float3 intrin_cosf(float3 x) { return make_float3(__cosf(x.x), __cosf(x.y), __cosf(x.z)); }

	__device__ float3 sin(float3 x) { return make_float3(sinf(x.x), sinf(x.y), sinf(x.z)); }
	__device__ float2 sin(float2 x) { return make_float2(sinf(x.x), sinf(x.y)); }
	__device__ float intrin_sinf(float x) { return __sinf(x); }

	__device__ float intrin_expf(float x) { return __expf(x); }
	__device__ float3 intrin_expf(float3 x) { return make_float3(__expf(x.x), __expf(x.y), __expf(x.z)); }
	__device__ float intrin_logf(float x) { return __logf(x); }

	__device__ float3 atan2(float3 y, float3 x) { return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z)); }

	__device__ float2 exp(float2 x) { return make_float2(expf(x.x), expf(x.y)); }
	__device__ float3 exp(float3 x) { return make_float3(expf(x.x), expf(x.y), expf(x.z)); }
	__device__ float intrin_expm1f(float x) { return hippt::intrin_expf(x) - 1.0f; }
	__device__ float3 ldexp(float3 x, int exp) { return make_float3(ldexpf(x.x, exp), ldexpf(x.y, exp), ldexpf(x.z, exp)); }

	// (exp(x) - 1)/x with cancellation of rounding errors.
	// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p. 19]
	__device__ float expm1_over_x_precise(const float x)
	{
		const float u = hippt::intrin_expf(x);

		if (u == 1.0f)
			return 1.0f;

		const float y = u - 1.0f;

		if (hippt::abs(x) < 1.0f)
			return y / hippt::intrin_logf(u);

		return y / x;
	}

	// Uses intrin_expm1f instead of the builtin expm1f()
	__device__ float expm1_over_x_fast(const float x)
	{
		float exp_x = hippt::intrin_expf(x);
		if (exp_x == 1.0f)
			return 1.0f;

		return (exp_x - 1.0f) / x;
	}

	__device__ float erfcf_fast(float x)
	{
		constexpr float TWO_OVER_ROOT_PI = 1.1283791670955125738961589031215f;
		constexpr float ERFC_SMALL = 0.0053854f;

		if (hippt::abs(x) < ERFC_SMALL)
			return 1.0f - TWO_OVER_ROOT_PI * x;

		float a, c, e, p, q, r, s;
		a = hippt::abs(x);
		c = hippt::min(a, 10.5f);
		s = -c * c;
		e = hippt::intrin_expf(s);
		q = 0.374177223624056f;
		p = -5.00032254520701E-05f;
		q = q * c + 1.29051354328887f;
		p = p * c + 0.212358010453875f;
		q = q * c + 1.84437448399707f;
		p = p * c + 0.715675302663111f;
		q = q * c + 1.0f;
		p = p * c + 1.0f;

		r = e / q;
		r = r * p;
		if (x < 0.0f)
			r = 2.0f - r;

		return r;
	}

	template <typename T>
	__device__ T square(T x) { return x * x; }

	__device__ float sqrt(float x) { return sqrtf(x); }
	__device__ float2 sqrt(float2 uv) { return make_float2(sqrtf(uv.x), sqrtf(uv.y)); }
	__device__ float3 sqrt(float3 uvw) { return make_float3(sqrtf(uvw.x), sqrtf(uvw.y), sqrtf(uvw.z)); }
	__device__ float rsqrt(float x) { return 1.0f / hippt::sqrt(x); }

	__device__ float pow_1_4(float x) { return sqrtf(sqrtf(x)); }
	__device__ constexpr float pow_3(float x) { return x * x * x; }
	__device__ constexpr float pow_4(float x) { float x2 = x * x; return x2 * x2; }
	__device__ constexpr float pow_5(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
	__device__ constexpr float pow_6(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x2; }

	__device__ float intrin_pow(float x, float y) { return __powf(x, y); }
	__device__ float pow_2_2_fit(float x) { return (exp2f(0.718151f * x) - 1.0f - 0.503456f * x) * 7.07342f; }

	__device__ float2 normalize(float2 u) { return u / sqrtf(hippt::dot(u, u)); }
	__device__ float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	__device__ bool is_nan(const T& v) { return isnan(v); }
	template <typename T>
	__device__ bool is_inf(const T& v) { return isinf(v); }
	__device__ bool is_zero(float x) { return x < NEAR_ZERO && x > -NEAR_ZERO; }

	__device__ unsigned int float_as_uint(float float_num) { return __float_as_uint(float_num); }
	__device__ float uint_as_float(unsigned int uint_num) { return __uint_as_float(uint_num); }

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

	/**
	 * The function returns the value at 'address' because the increment
	 */
	template <typename T>
	__device__ T atomic_fetch_add(T* address, T increment) { return atomicAdd(address, increment); }

	template <>
	__device__ unsigned char atomic_fetch_add(unsigned char* address, unsigned char increment)
	{
		// From https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars/59329536#59329536

		// offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
		size_t long_address_modulo = (size_t)address & 3;
		// the 32-bit address that overlaps the same memory
		unsigned int* base_address = (unsigned int*)((unsigned char*)address - long_address_modulo);
		// A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
		// The "4" signifies the position where the first byte of the second argument will end up in the output.
		unsigned int selectors[] = { 0x3214, 0x3240, 0x3410, 0x4210 };
		// for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
		unsigned int selector = selectors[long_address_modulo];
		unsigned int long_old, long_assumed, long_val, replacement;

		long_old = *base_address;

		do 
		{
			long_assumed = long_old;
			// replace bits in long_old that pertain to the char address with those from val
			long_val = __byte_perm(long_old, 0, long_address_modulo) + increment;
			replacement = __byte_perm(long_old, long_val, selector);
			long_old = atomicCAS(base_address, long_assumed, replacement);
		} while (long_old != long_assumed);

		return __byte_perm(long_old, 0, long_address_modulo);
	}

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

	template <>
	__device__ float atomic_compare_exchange(float* p, float cmp, float val) { return __int_as_float(atomicCAS((int*)p, __float_as_int(cmp), __float_as_int(val))); }

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
	__device__ float inverse_lerp(T value, T a, T b)
	{
		// Clamping
		value = hippt::max(a, hippt::min(value, b));

		return (value - a) / (b - a);
	}

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
	/**
	 * Returns a bit mask whose bits are set to 1 for threads that evaluated the predicate to true.
	 */
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

	__device__ void debugbreak() { }
	
	__device__ float idx(float3 v, int index) { return *(&v.x + index); }

#else
	constexpr float M_INV_TWO_PI = 0.15915494309189533577f;	// 1.0f / (2.0f * M_PI)
	constexpr float M_INV_PI = 0.31830988618379067154f;	// 1.0f / M_PI
	constexpr float M_PI_TWO = 1.57079632679489661923;	// pi/2
	constexpr float M_Pi = 3.14159265358979323846;	// pi
	constexpr float M_TWO_PI = 6.28318530717958647693f;	// 2.0f * M_PI
	constexpr float M_FOUR_PI = 12.5663706143591729539f;	// 4.0f * M_PI
	constexpr float M_TWO_PI_SQUARED = 19.73920880217871723767f;	// 2.0f * M_PI ^ 2
	constexpr float NEAR_ZERO = 1.0e-10f;

	constexpr float FLOAT_MAX = 3.402823466e+38f;
	constexpr float FLOAT_MIN = 1.175494351e-38f;
	constexpr float FLOAT_EPSILON = 1.192092896e-07f;

	constexpr float Infinity() { return ((float)(1e+300)); }

	/**
	 * Returns the 'warpSize' runtime constant of the GPU
	 */
	static int warp_size() { return 1; }
	static int thread_idx_x() { return 0; }
	static int thread_idx_y() { return 0; }
	static int thread_idx_global() { return 0; }
	static bool is_pixel_index(int x, int y) { return false; }
	static int current_warp_lane() { return 0; }

	template <typename T>
	static T ldg_load(T* address) { return *address; }

	static float3 cross(float3 u, float3 v) { return hiprt::cross(u, v); }
	static float dot(float3 u, float3 v) { return hiprt::dot(u, v); }
	static float dot(float2 u, float2 v) { return u.x * v.x + u.y * v.y; }

	static float length(float3 u) { return sqrtf(dot(u, u)); }
	static float length2(float3 u) { return dot(u, u); }

	static float3 abs(float3 u) { return make_float3(std::abs(u.x), std::abs(u.y), std::abs(u.z)); }
	static float abs(float a) { return std::abs(a); }

	static float fma(float a, float b, float c) { return a * b + c; }
	static float2 fma(float2 a, float2 b, float2 c) { return make_float2(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y)); }
	static float3 fma(float3 a, float3 b, float3 c) { return make_float3(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y), hippt::fma(a.z, b.z, c.z)); }





	template <typename T>
	static T max(T a, T b) { return a > b ? a : b; }
	/**
	 * Component-wise max of float3 and int3
	 */
	template <>
	float3 max(float3 a, float3 b) { return make_float3(hiprt::max(a.x, b.x), hiprt::max(a.y, b.y), hiprt::max(a.z, b.z)); }
	template <>
	int3 max(int3 a, int3 b) { return make_int3(hiprt::max(a.x, b.x), hiprt::max(a.y, b.y), hiprt::max(a.z, b.z)); }




	template <typename T>
	static T min(T a, T b) { return a < b ? a : b; }

	/**
	 * Component-wise min of float3 and int3
	 */
	template <>
	float3 min(float3 a, float3 b) { return make_float3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }
	template <>
	int3 min(int3 a, int3 b) { return make_int3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z)); }





	/**
	 * Minimum of each component of the float3 against x
	 */
	static float3 min(float3 a, float x) { return make_float3(hiprt::min(a.x, x), hiprt::min(a.y, x), hiprt::min(a.z, x)); }
	static float3 min(float x, float3 a) { return hippt::min(a, x); }

	/**
	 * Minimum of each component of the mat 2x2 against x
	 */
	static float2x2 min(float x, float2x2 a)  { return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x)); }
	static float2x2 min(float2x2 a, float x)  { return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x)); }

	template <typename T>
	static T clamp(T min_val, T max_val, T val) { return hiprt::min(max_val, hiprt::max(min_val, val)); }

	static float2 cos(float2 x) { return make_float2(std::cos(x.x), std::cos(x.y)); }
	static float3 cos(float3 x) { return make_float3(std::cos(x.x), std::cos(x.y), std::cos(x.z)); }
	static float intrin_cosf(float x) { return std::cos(x); }
	static float3 intrin_cosf(float3 x) { return make_float3(std::cos(x.x), std::cos(x.y), std::cos(x.z)); }

	static float2 sin(float2 x) { return make_float2(std::sin(x.x), std::sin(x.y)); }
	static float3 sin(float3 x) { return make_float3(std::sin(x.x), std::sin(x.y), std::sin(x.z)); }
	static float intrin_sinf(float x) { return std::sin(x); }

	static float intrin_expf(float x) { return expf(x); }
	static float3 intrin_expf(float3 x) { return make_float3(expf(x.x), expf(x.y), expf(x.z)); }
	static float intrin_logf(float x) { return logf(x); }

	static float3 atan2(float3 y, float3 x) { return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z)); }

	static float2 exp(float2 x) { return make_float2(expf(x.x), expf(x.y)); }
	static float3 exp(float3 x) { return make_float3(expf(x.x), expf(x.y), expf(x.z)); }
	static float intrin_expm1f(float x) { return hippt::intrin_expf(x) - 1.0f; }
	static float3 ldexp(float3 x, int exp) { return make_float3(std::ldexp(x.x, exp), std::ldexp(x.y, exp), std::ldexp(x.z, exp)); }

	// (exp(x) - 1)/x with cancellation of rounding errors.
	// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p. 19]
	static float expm1_over_x_precise(const float x)
	{
		const float u = hippt::intrin_expf(x);

		if (u == 1.0f)
			return 1.0f;

		const float y = u - 1.0f;

		if (hippt::abs(x) < 1.0f)
			return y / hippt::intrin_logf(u);

		return y / x;
	}

	static float expm1_over_x_fast(const float x)
	{
		float exp_x = hippt::intrin_expf(x);
		if (exp_x == 1.0f)
			return 1.0f;

		return (exp_x - 1.0f) / x;
	}

	static float erfcf_fast(float x)
	{
		constexpr float TWO_OVER_ROOT_PI = 1.1283791670955125738961589031215f;
		constexpr float ERFC_SMALL = 0.0053854f;

		if (hippt::abs(x) < ERFC_SMALL)
			return 1.0f - TWO_OVER_ROOT_PI * x;

		float a, c, e, p, q, r, s;
		a = hippt::abs(x);
		c = hippt::min(a, 10.5f);
		s = -c * c;
		e = hippt::intrin_expf(s);
		q = 0.374177223624056f;
		p = -5.00032254520701E-05f;
		q = q * c + 1.29051354328887f;
		p = p * c + 0.212358010453875f;
		q = q * c + 1.84437448399707f;
		p = p * c + 0.715675302663111f;
		q = q * c + 1.0f;
		p = p * c + 1.0f;

		r = e / q;
		r = r * p;
		if (x < 0.0f)
			r = 2.0f - r;

		return r;
	}

	template <typename T>
	static T square(T x) { return x * x; }

	static float sqrt(float x) { return sqrtf(x); }
	static float2 sqrt(float2 uv) { return make_float2(sqrtf(uv.x), sqrtf(uv.y)); }
	static float3 sqrt(float3 uvw) { return make_float3(sqrtf(uvw.x), sqrtf(uvw.y), sqrtf(uvw.z)); }
	static float rsqrt(float x) { return 1.0f / sqrtf(x); }

	static float pow_1_4(float x) { return sqrtf(sqrtf(x)); }
	static constexpr float pow_3(float x) { return x * x * x; }
	static constexpr float pow_4(float x) { float x2 = x * x; return x2 * x2; }
	static constexpr float pow_5(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x; }
	static constexpr float pow_6(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x2; }

	static float intrin_pow(float x, float y) { return powf(x, y); }
	static float pow_2_2_fit(float x) { return (exp2f(0.718151f * x) - 1.0f - 0.503456f * x) * 7.07342f; }

	static float2 normalize(float2 u) { return u / sqrtf(hippt::dot(u, u)); }
	static float3 normalize(float3 u) { return hiprt::normalize(u); }

	template <typename T>
	static bool is_nan(const T& v) { return std::isnan(v); }
	template <typename T>
	static constexpr bool is_inf(const T& v) { return std::isinf(v); }
	static bool is_zero(float x) { return x < NEAR_ZERO && x > -NEAR_ZERO; }

	static unsigned int float_as_uint(float float_num) { return std::bit_cast<unsigned int>(float_num);}
	static float uint_as_float(unsigned int uint_num) { return std::bit_cast<float>(uint_num); }

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

	/**
	 * The function returns the value at 'address' because the increment
	 */
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
	 * 
	 * The function returns old (Compare And Swap).
	 */
	template <typename T>
	T atomic_compare_exchange(std::atomic<T>* atomic_address, T expected, T new_value)
	{
		atomic_address->compare_exchange_strong(expected, new_value);
			
		return expected;
	}

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	static T lerp(T a, T b, float t) { return (1.0f - t) * a + t * b; }

	/**
	 * For a 'value' between 'a' and 'b', returns 't' such that
	 * (1.0f - t) * a + t * b = value
	 *
	 * For 'value' == 'a', returns 0.0f
	 * For 'value' == 'b', returns 1.0f
	 */
	template <typename T>
	static float inverse_lerp(T value, T a, T b) 
	{ 
		// Clamping
		value = hippt::max(a, hippt::min(value, b)); 
		
		return (value - a) / (b - a); 
	}
	
	/**
	 * Reference: https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml
	 * 
	 * For t == min, returns 0.0f
	 * For t == max, returns 1.0f
	 * Smoothstep interpolation in between
	 */
	template <typename T>
	static T smoothstep(T min, T max, float x) 
	{ 
		float t = hippt::clamp(0.0f, 1.0f, (x - min) / (max - min));

		return t * t * (3.0f - 2.0f * t);
	}

	static float fract(float a) { return a - floorf(a); }

	template <typename T>
	static int popc(T bitmask) { return std::popcount(bitmask); }

	/**
	 * Finds the position of least signigicant bit set to 1 in a 32 bit unsigned integer.
	 * Returs a value between 0 and 32 inclusive.
	 *
	 * Returns 0 if all bits are zero
	 */
	static unsigned int ffs(unsigned int bitmask)
	{
		for (int i = 0; i < sizeof(unsigned int) * 8; i++)
			if (bitmask & (1 << i))
				return i;

		return 0;
	}

	static bool warp_any(unsigned int thread_mask, bool predicate) { return predicate; }
	/**
	 * Returns a bit mask whose bits are set to 1 for threads that evaluated the predicate to true.
	 */
	static unsigned long long int warp_ballot(unsigned int thread_mask, bool predicate) { return predicate ? 1 : 0; }
	static unsigned int warp_activemask() { return 1; }

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
	static T warp_shfl(T var, int srcLane, int width = 1) { return var; }

	/**
	 * Returns the index within its warp (not group) of the calling thread
	 * 
	 * Warp sizes of 1 on the CPU
	 */
	static unsigned int warp_2D_thread_index() { return 1; }

	static void debugbreak() { Debug::debugbreak(); }

	static float idx(float3 v, int index) { return *(&v.x + index); }
#endif
}

HIPRT_DEVICE static float3 matrix_X_point(const float4x4& m, const float3& p)
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

HIPRT_DEVICE static float3 matrix_X_vec(const float3x3& m, const float3& u)
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

HIPRT_DEVICE static float3 matrix_X_vec(const float4x4& m, const float3& u)
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

HIPRT_DEVICE static float2x2 operator+(const float2x2& a, const float2x2& b)
{
	float2x2 result;

	result.m[0][0] = a.m[0][0] + b.m[0][0];
	result.m[0][1] = a.m[0][1] + b.m[0][1];
	result.m[1][0] = a.m[1][0] + b.m[1][0];
	result.m[1][1] = a.m[1][1] + b.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator-(const float2x2& a, const float2x2& b)
{
	float2x2 result;

	result.m[0][0] = a.m[0][0] - b.m[0][0];
	result.m[0][1] = a.m[0][1] - b.m[0][1];
	result.m[1][0] = a.m[1][0] - b.m[1][0];
	result.m[1][1] = a.m[1][1] - b.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator*(const float2x2& a, const float2x2& b)
{
	float2x2 result;

	result.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0];
	result.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1];
	result.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0];
	result.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator*(const float k, const float2x2& a)
{
	float2x2 result;

	result.m[0][0] = k * a.m[0][0];
	result.m[0][1] = k * a.m[0][1];
	result.m[1][0] = k * a.m[1][0];
	result.m[1][1] = k * a.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator*(const float2x2& a, const float k)
{
	return k * a;
}

HIPRT_DEVICE static float2 operator*(const float2x2& a, const float2& v)
{
	float2 result;

	result.x = a.m[0][0] * v.x + a.m[0][1] * v.y;
	result.y = a.m[1][0] * v.x + a.m[1][1] * v.y;

	return result;
}

HIPRT_DEVICE static float2x2 operator/(const float2x2& a, const float k)
{ 
	float inv_k = 1.0f / k;
	float2x2 result;

	result.m[0][0] = a.m[0][0] * inv_k;
	result.m[0][1] = a.m[0][1] * inv_k;
	result.m[1][0] = a.m[1][0] * inv_k;
	result.m[1][1] = a.m[1][1] * inv_k;

	return result;
}

HIPRT_DEVICE static float2x2 transpose(const float2x2& m)
{
	float2x2 result;

	result.m[0][0] = m.m[0][0];
	result.m[0][1] = m.m[1][0];
	result.m[1][0] = m.m[0][1];
	result.m[1][1] = m.m[1][1];

	return result;
}

HIPRT_DEVICE static float determinant(const float2x2& m)
{
	return m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];
}

HIPRT_DEVICE static float2x2 outer_product(const float2& a, const float2& b) 
{
	float2x2 out;

	// row 0
	out.m[0][0] = a.x * b.x;
	out.m[0][1] = a.x * b.y;
	// row 1
	out.m[1][0] = a.y * b.x;
	out.m[1][1] = a.y * b.y;

	return out;
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
