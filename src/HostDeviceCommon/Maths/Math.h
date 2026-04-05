/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATH_H
#define HOST_DEVICE_COMMON_MATH_H

#if defined(__KERNELCC__)
#include <hiprt/hiprt_device.h>
#else
#include "HostDeviceCommon/Maths/VecTypes.h"

// For hippt::debugbreak()
#include "Utils/Debug.h"
#endif

#if !defined(__KERNELCC__) || defined(HIPRT_BITCODE_LINKING)
// For std::atomic in hippt::
#include <atomic>
// For std::bit_cast in hippt::
#include <bit>
#endif

#include "HostDeviceCommon/AtomicType.h"

#include "HostDeviceCommon/Maths/Float2x2.h"
#include "HostDeviceCommon/Maths/Float3x3.h"
#include "HostDeviceCommon/Maths/Float4x4.h"
#include "HostDeviceCommon/Maths/Types.h"

// Here we're defining aliases for common functions used in shader code.
//
// Because the same shader code can be used both on the CPU and the GPU,
// both code have to compile either through the classical C++ compiler or
// through the GPU shader compiler. This means that we have to use functions
// that were meant to be used on the CPU or on the GPU (depending on the case).
namespace hippt
{
#ifdef __KERNELCC__
	constexpr float M_INV_TWO_PI	 = 0.15915494309189533577f;	 // 1.0f / (2.0f * M_PI)
	constexpr float M_INV_PI		 = 0.31830988618379067154f;	 // 1.0f / M_PI
	constexpr float M_PI_TWO		 = 1.57079632679489661923f;	 // pi/2
	constexpr float M_Pi			 = 3.14159265358979323846f;	 // pi
	constexpr float M_TWO_PI		 = 6.28318530717958647693f;	 // 2.0f * M_PI
	constexpr float M_FOUR_PI		 = 12.5663706143591729539f;	 // 4.0f * M_PI
	constexpr float M_TWO_PI_SQUARED = 19.73920880217871723767f; // 2.0f * M_PI ^ 2
	constexpr float NEAR_ZERO		 = 1.0e-10f;

	constexpr float FLOAT_MAX	  = 3.402823466e+38f;
	constexpr float FLOAT_MIN	  = 1.175494351e-38f;
	constexpr float FLOAT_EPSILON = 1.192092896e-07f;

	__device__ static float Infinity()
	{
		return __int_as_float(0x7f800000);
	}

	/**
	 * Returns the 'warpSize' runtime constant of the GPU
	 */
	__device__ static int warp_size()
	{
		return warpSize;
	}

	__device__ static unsigned int thread_idx_x()
	{
		return threadIdx.x + blockIdx.x * blockDim.x;
	}

	__device__ static unsigned int thread_idx_y()
	{
		return threadIdx.y + blockIdx.y * blockDim.y;
	}

	__device__ static unsigned int thread_idx_global()
	{
		return hippt::thread_idx_x() + hippt::thread_idx_y() * blockDim.x * gridDim.x;
	}

	__device__ static bool is_pixel_index(int x, int y)
	{
		return hippt::thread_idx_x() == x && hippt::thread_idx_y() == y;
	}

	__device__ static int current_warp_lane()
	{
		return (threadIdx.x + threadIdx.y * blockDim.x) % hippt::warp_size();
	}

	template <typename T>
	__device__ static T ldg_load(T* address)
	{
		return __ldg(address);
	}

	__device__ static float3_t cross(float3_t u, float3_t v)
	{
		return hiprt::cross(u, v);
	}

	__device__ static float dot(float3_t u, float3_t v)
	{
		return hiprt::dot(u, v);
	}

	__device__ static float dot(float2_t u, float2_t v)
	{
		return u.x * v.x + u.y * v.y;
	}

	__device__ static float sqrt(float x)
	{
		return sqrtf(x);
	}

	__device__ static float2_t sqrt(float2_t uv)
	{
		return make_float2(hippt::sqrt(uv.x), hippt::sqrt(uv.y));
	}

	__device__ static float3_t sqrt(float3_t uvw)
	{
		return make_float3(hippt::sqrt(uvw.x), hippt::sqrt(uvw.y), hippt::sqrt(uvw.z));
	}

	__device__ static float rsqrt(float x)
	{
		return 1.0f / hippt::sqrt(x);
	}

	__device__ static float length(float3_t u)
	{
		return hippt::sqrt(hippt::dot(u, u));
	}

	__device__ static float length(float2_t u)
	{
		return hippt::sqrt(hippt::dot(u, u));
	}

	__device__ static float length2(float3_t u)
	{
		return hippt::dot(u, u);
	}

	__device__ static float3_t abs(float3_t u)
	{
		return make_float3(fabsf(u.x), fabsf(u.y), fabsf(u.z));
	}

	__device__ static float abs(float a)
	{
		return fabsf(a);
	}

	/**
	 * a * b + c
	 */
	__device__ static float fma(float a, float b, float c)
	{
		return fmaf(a, b, c);
	}

	__device__ static float2_t fma(float2_t a, float2_t b, float2_t c)
	{
		return make_float2(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y));
	}

	__device__ static float3_t fma(float3_t a, float3_t b, float3_t c)
	{
		return make_float3(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y), hippt::fma(a.z, b.z, c.z));
	}

	__device__ float mix_fma(float x, float y, float a)
	{
		return hippt::fma(a, y, hippt::fma(-a, x, x));
	}

	template <typename T>
	__device__ static T max(T a, T b)
	{
		return a > b ? a : b;
	}

	/**
	 * Component-wise max of float3_t and int3_t
	 */
	template <>
	__device__ float3_t max(float3_t a, float3_t b)
	{
		return make_float3(hiprt::max(a.x, b.x), hiprt::max(a.y, b.y), hiprt::max(a.z, b.z));
	}

	template <>
	__device__ int3_t max(int3_t a, int3_t b)
	{
		return make_int3(hiprt::max(a.x, b.x), hiprt::max(a.y, b.y), hiprt::max(a.z, b.z));
	}

	template <typename T>
	__device__ static T min(T a, T b)
	{
		return a < b ? a : b;
	}

	/**
	 * Component-wise min of float3_t and int3_t
	 */
	template <>
	__device__ float3_t min(float3_t a, float3_t b)
	{
		return make_float3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z));
	}

	template <>
	__device__ int3_t min(int3_t a, int3_t b)
	{
		return make_int3(hiprt::min(a.x, b.x), hiprt::min(a.y, b.y), hiprt::min(a.z, b.z));
	}

	/**
	 * Minimum of each component of the float3_t against x
	 */
	__device__ static float3_t min(float3_t a, float x)
	{
		return make_float3(hiprt::min(a.x, x), hiprt::min(a.y, x), hiprt::min(a.z, x));
	}

	__device__ static float3_t min(float x, float3_t a)
	{
		return hippt::min(a, x);
	}

	/**
	 * Minimum of each component of the mat 2x2 against x
	 */
	__device__ static float2x2 min(float x, float2x2 a)
	{
		return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x));
	}

	__device__ static float2x2 min(float2x2 a, float x)
	{
		return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x));
	}

	template <typename T>
	__device__ static T clamp(T min_val, T max_val, T val)
	{
		return hiprt::min(max_val, hiprt::max(min_val, val));
	}

	__device__ static float max(float a, float b)
	{
		return a > b ? a : b;
	}

	__device__ static float min(float a, float b)
	{
		return a < b ? a : b;
	}

	__device__ static float clamp(float min_val, float max_val, float val)
	{
		return hiprt::clamp(val, min_val, max_val);
	}

	__device__ static float3_t cos(float3_t x)
	{
		return make_float3(cosf(x.x), cosf(x.y), cosf(x.z));
	}

	__device__ static float2_t cos(float2_t x)
	{
		return make_float2(cosf(x.x), cosf(x.y));
	}

	__device__ static float intrin_cosf(float x)
	{
		return __cosf(x);
	}

	__device__ static float3_t intrin_cosf(float3_t x)
	{
		return make_float3(hippt::intrin_cosf(x.x), hippt::intrin_cosf(x.y), hippt::intrin_cosf(x.z));
	}

	__device__ static float3_t sin(float3_t x)
	{
		return make_float3(sinf(x.x), sinf(x.y), sinf(x.z));
	}

	__device__ static float2_t sin(float2_t x)
	{
		return make_float2(sinf(x.x), sinf(x.y));
	}

	__device__ static float intrin_sinf(float x)
	{
		return __sinf(x);
	}

	__device__ static float3_t atan2(float3_t y, float3_t x)
	{
		return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z));
	}

	__device__ static float2_t exp(float2_t x)
	{
		return make_float2(expf(x.x), expf(x.y));
	}

	__device__ static float3_t exp(float3_t x)
	{
		return make_float3(expf(x.x), expf(x.y), expf(x.z));
	}

	__device__ static float intrin_expf(float x)
	{
		return __expf(x);
	}

	__device__ static float3_t intrin_expf(float3_t x)
	{
		return make_float3(hippt::intrin_expf(x.x), hippt::intrin_expf(x.y), hippt::intrin_expf(x.z));
	}

	__device__ static float intrin_expm1f(float x)
	{
		return hippt::intrin_expf(x) - 1.0f;
	}

	__device__ static float3_t ldexp(float3_t x, int exp)
	{
		return make_float3(ldexpf(x.x, exp), ldexpf(x.y, exp), ldexpf(x.z, exp));
	}

	__device__ static float intrin_logf(float x)
	{
		return __logf(x);
	}

	// (exp(x) - 1)/x with cancellation of rounding errors.
	// [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p. 19]
	__device__ static float expm1_over_x_precise(const float x)
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
	__device__ static float expm1_over_x_fast(const float x)
	{
		float exp_x = hippt::intrin_expf(x);
		if (exp_x == 1.0f)
			return 1.0f;

		return (exp_x - 1.0f) / x;
	}

	__device__ static float erfcf_fast(float x)
	{
		constexpr float TWO_OVER_ROOT_PI = 1.1283791670955125738961589031215f;
		constexpr float ERFC_SMALL		 = 0.0053854f;

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
	__device__ static T square(T x)
	{
		return x * x;
	}

	__device__ static float pow_1_4(float x)
	{
		return hippt::sqrt(hippt::sqrt(x));
	}

	__device__ static constexpr float pow_3(float x)
	{
		return x * x * x;
	}

	__device__ static constexpr float pow_4(float x)
	{
		float x2 = x * x;
		return x2 * x2;
	}

	__device__ static constexpr float pow_5(float x)
	{
		float x2 = x * x;
		float x4 = x2 * x2;
		return x4 * x;
	}

	__device__ static constexpr float pow_6(float x)
	{
		float x2 = x * x;
		float x4 = x2 * x2;
		return x4 * x2;
	}

	__device__ static float intrin_pow(float x, float y)
	{
		return __powf(x, y);
	}

	__device__ static float pow_2_2_fit(float x)
	{
		return (exp2f(0.718151f * x) - 1.0f - 0.503456f * x) * 7.07342f;
	}

	__device__ static float2_t normalize(float2_t u)
	{
		return u / hippt::sqrt(hippt::dot(u, u));
	}

	__device__ static float3_t normalize(float3_t u)
	{
		return hiprt::normalize(u);
	}

	template <typename T>
	__device__ static bool is_nan(const T& v)
	{
		return isnan(v);
	}

	template <typename T>
	__device__ static bool is_inf(const T& v)
	{
		return isinf(v);
	}

	__device__ static bool is_zero(float x)
	{
		return x < NEAR_ZERO && x > -NEAR_ZERO;
	}

	__device__ static bool is_finite(float x)
	{
		return isfinite(x);
	}

	__device__ static unsigned int float_as_uint(float float_num)
	{
		return __float_as_uint(float_num);
	}

	__device__ static float uint_as_float(unsigned int uint_num)
	{
		return __uint_as_float(uint_num);
	}

	__device__ float fp16_bits_to_fp32(unsigned short int half_bits)
	{
		// Source: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion

		const unsigned int e = (half_bits & 0x7C00) >> 10;			 // exponent
		const unsigned int m = (half_bits & 0x03FF) << 13;			 // mantissa
		const unsigned int v = hippt::float_as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format

		return hippt::uint_as_float((half_bits & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
									((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
	}

	// IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	__device__ unsigned short int fp32_to_fp16_bits(const float x)
	{
		// Source: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion

		const unsigned int b = hippt::float_as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
		const unsigned int e = (b & 0x7F800000) >> 23;				 // exponent
		const unsigned int m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
		return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
			   ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
	}

	__device__ static unsigned short int half_as_ushort(fp16 half)
	{
		return __half_as_ushort(half);
	}

	/**
	 * Reads the 32-bit or 64-bit word old located at the address 'address'
	 * in global or shared memory and stores 'value' to memory at the same address.
	 *
	 * These two operations are performed in one atomic transaction.
	 * The function returns old.
	 */
	template <typename T>
	__device__ static T atomic_exchange(T* address, T value)
	{
		return atomicExch(address, value);
	}

	/**
	 * Reads the 32-bit or 64-bit word 'old' located at 'address' in global or shared memory,
	 * computes the maximum of 'old' and 'value', and stores the result back to memory at the
	 * same address.
	 *
	 * The function returns 'old'
	 */
	template <typename T>
	__device__ static T atomic_max(T* address, T value)
	{
		return atomicMax(address, value);
	}

	/**
	 * Reads the 32-bit or 64-bit word 'old' located at 'address' in global or shared memory,
	 * computes the minimum of 'old' and 'value', and stores the result back to memory at the
	 * same address.
	 *
	 * The function returns 'old'
	 */
	template <typename T>
	__device__ static T atomic_min(T* address, T value)
	{
		return atomicMin(address, value);
	}

	/**
	 * The function returns the value at 'address' because the increment
	 */
	template <typename T>
	__device__ static T atomic_fetch_add(T* address, T increment)
	{
		return atomicAdd(address, increment);
	}

	template <typename T>
	__device__ static T atomic_fetch_add_gpu(T* address, T increment)
	{
		return atomic_fetch_add(address, increment);
	}

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
			long_val	= __byte_perm(long_old, 0, long_address_modulo) + increment;
			replacement = __byte_perm(long_old, long_val, selector);
			long_old	= atomicCAS(base_address, long_assumed, replacement);
		} while (long_old != long_assumed);

		return __byte_perm(long_old, 0, long_address_modulo);
	}

	template <>
	__device__ short int atomic_fetch_add(short int* address, short int val)
	{
		// Source: https://forums.developer.nvidia.com/t/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/22712/11

		unsigned int* base_address = (unsigned int*)((size_t)address & ~2);
		unsigned int long_val	   = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
		unsigned int long_old	   = atomicAdd(base_address, long_val);

		if ((size_t)address & 2)
			return (short)(long_old >> 16);
		else
		{
			unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

			if (overflow)
				atomicSub(base_address, overflow);

			return (short)(long_old & 0xffff);
		}
	}

	template <typename T>
	__device__ static T atomic_load(T* address)
	{
		return *address;
	}

	/**
	 * Reads the 16/32/64 bit word at the 'address' in global or shared memory,
	 * computes(*address == expected ? new_value : *address), and stores the result
	 * back to memory at the same address.
	 *
	 * These three operations are performed in one atomic transaction.
	 * The function returns old (Compare And Swap).
	 */
	template <typename T>
	__device__ static T atomic_compare_exchange(T* address, T expected, T new_value)
	{
		return atomicCAS(address, expected, new_value);
	}

	template <>
	__device__ float atomic_compare_exchange(float* p, float cmp, float val)
	{
		return __int_as_float(atomicCAS((int*)p, __float_as_int(cmp), __float_as_int(val)));
	}

	template <typename T>
	__device__ static T atomic_compare_exchange_gpu(T* address, T expected, T new_value)
	{
		return atomic_compare_exchange<T>(address, expected, new_value);
	}

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	__device__ static T lerp(T a, T b, float t)
	{
		return (1.0f - t) * a + t * b;
	}

	/**
	 * For a 'value' between 'a' and 'b', returns 't' such that
	 * (1.0f - t) * a + t * b = value
	 *
	 * For 'value' == 'a', returns 0.0f
	 * For 'value' == 'b', returns 1.0f
	 */
	template <typename T>
	__device__ static float inverse_lerp(T value, T a, T b)
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
	__device__ static T smoothstep(T min, T max, float x)
	{
		float t = hippt::clamp(0.0f, 1.0f, (x - min) / (max - min));

		return t * t * (3.0f - 2.0f * t);
	}

	__device__ static float fract(float a)
	{
		return a - floorf(a);
	}

	template <typename T>
	__device__ unsigned int popc(T bitmask)
	{
		return 0;
	}

	template <>
	__device__ unsigned int popc(unsigned int bitmask)
	{
		return __popc(bitmask);
	}

	template <>
	__device__ unsigned int popc(unsigned long long int bitmask)
	{
		return __popcll(bitmask);
	}

	/**
	 * Finds the position of least signigicant bit set to 1 in a 32 bit unsigned integer.
	 * Returs a value between 0 and 32 inclusive.
	 *
	 * Returns 0 if all bits are zero
	 */
	__device__ static int ffs(unsigned int bitmask)
	{
		return __ffs(bitmask);
	}

	template <typename T>
	__device__ static T clz(T bitmask)
	{
		return __clz(bitmask);
	}

	// TODO these functions require __sync on modern NVIDIA GPUs. We should check that with __CUDACC__
	__device__ static bool warp_any(unsigned int thread_mask, bool predicate)
	{
		return __any(predicate);
	}

	/**
	 * Returns a bit mask whose bits are set to 1 for threads that evaluated the predicate to true.
	 */
	__device__ static unsigned long long int warp_ballot(unsigned int thread_mask, bool predicate)
	{
		return __ballot(predicate);
	}

	__device__ static unsigned long long int warp_activemask()
	{
		return hippt::warp_ballot(0xFFFFFFFF, true);
	}

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
	__device__ static T warp_shfl(T var, int src_lane, int width = warpSize)
	{
#ifdef __CUDACC__
		return __shfl_sync(0xFFFFFFFF, var, src_lane, width);
#else
		return __shfl(var, src_lane, width);
#endif
	}

	/**
	 * T can be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type.
	 * The warp shuffle functions exchange values between threads within a warp.
	 *
	 * The optional width argument specifies subgroups, in which the warp can be divided to share the variables. It has to be a power of two smaller than or
	 * equal to warpSize. If it is smaller than warpSize, the warp is grouped into separate groups, that are each indexed from 0 to width as if it was its own
	 * entity, and only the lanes within that subgroup participate in the shuffle. The lane indices in the subgroup are given by laneIdx % width.
	 *
	 * The 64-bit unsigned integer mask argument specifies the lanes of the warp that will participate. Each participating thread must have its own bit set in
	 * its mask argument, and all active threads specified in any mask argument must execute the same call with the same mask, otherwise the result is
	 * undefined. The implementation includes a static assert to check that the program source uses the correct type for the mask.
	 */
	template <typename T>
	__device__ static T warp_shfl_sync(unsigned long long int mask, T var, int srcLane, int width = warpSize)
	{
		return __shfl_sync(mask, var, srcLane, width);
	}

	/**
	 * Copy from a lane with higher ID relative to caller
	 */
	template <typename T>
	__device__ T warp_shfl_down(T var, int delta, int width = warpSize)
	{
#ifdef __CUDACC__
		return __shfl_down_sync(0xFFFFFFFF, var, delta, width);
#else
		return __shfl_down(var, delta, width);
#endif
	}

	/**
	 * Copy from a lane with lower ID relative to caller
	 */
	template <typename T>
	__device__ T warp_shfl_up(T var, int delta, int width = warpSize)
	{
#ifdef __CUDACC__
		return __shfl_up_sync(0xFFFFFFFF, var, delta, width);
#else
		return __shfl_up(var, delta, width);
#endif
	}

	template <typename T>
	__device__ T warp_reduce_max(unsigned long long int thread_mask, T variable)
	{
#ifdef __CUDACC__
		return __reduce_max_sync(static_cast<unsigned int>(thread_mask & 0xFFFFFFFF), variable);
#else
		for (int offset = warpSize / 2; offset > 0; offset >>= 1)
			variable = max(variable, __shfl_down(variable, offset));

		return variable;
#endif
	}

	__device__ void syncwarp(unsigned int mask)
	{
#ifdef __CUDACC__
		__syncwarp(mask);
#endif
	}

	/**
	 * Returns the index within its warp (not group) of the calling thread
	 */
	__device__ static unsigned int warp_2D_thread_index()
	{
		// warpSize assuming to be a power of 2 so the '&' operation
		// here is a modulo
		return (threadIdx.x + threadIdx.y * blockDim.x) & warpSize;
	}

	__device__ static void debugbreak() {}

	__device__ static float idx(float3_t v, int index)
	{
		return *(&v.x + index);
	}

#else
	constexpr float M_INV_TWO_PI	 = 0.15915494309189533577f;	 // 1.0f / (2.0f * M_PI)
	constexpr float M_INV_PI		 = 0.31830988618379067154f;	 // 1.0f / M_PI
	constexpr float M_PI_TWO		 = 1.57079632679489661923;	 // pi/2
	constexpr float M_Pi			 = 3.14159265358979323846;	 // pi
	constexpr float M_TWO_PI		 = 6.28318530717958647693f;	 // 2.0f * M_PI
	constexpr float M_FOUR_PI		 = 12.5663706143591729539f;	 // 4.0f * M_PI
	constexpr float M_TWO_PI_SQUARED = 19.73920880217871723767f; // 2.0f * M_PI ^ 2
	constexpr float NEAR_ZERO		 = 1.0e-10f;

	constexpr float FLOAT_MAX	  = 3.402823466e+38f;
	constexpr float FLOAT_MIN	  = 1.175494351e-38f;
	constexpr float FLOAT_EPSILON = 1.192092896e-07f;

	constexpr float Infinity()
	{
		return ((float)(1e+300));
	}

	/**
	 * Returns the 'warpSize' runtime constant of the GPU
	 */
	static constexpr int warp_size()
	{
		return 1;
	}

	static unsigned int thread_idx_x()
	{
		return 0u;
	}

	static unsigned int thread_idx_y()
	{
		return 0u;
	}

	static unsigned int thread_idx_global()
	{
		return 0u;
	}

	static bool is_pixel_index(int x, int y)
	{
		return true;
	}

	static int current_warp_lane()
	{
		return 0;
	}

	template <typename T>
	static T ldg_load(T* address)
	{
		return *address;
	}

	static float3_t cross(float3_t u, float3_t v)
	{
		return make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
	}

	/*static float3_t cross(hiprtFloat3 u, float3_t v) { return make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x); }

static
	 * float3_t cross(float3_t u, hiprtFloat3 v) { return make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x); }*/
	static float dot(float3_t u, float3_t v)
	{
		return u.x * v.x + u.y * v.y + u.z * v.z;
	}

	static float dot(float2_t u, float2_t v)
	{
		return u.x * v.x + u.y * v.y;
	}

	static float sqrt(float x)
	{
		return sqrtf(x);
	}

	static float2_t sqrt(float2_t uv)
	{
		return make_float2(hippt::sqrt(uv.x), hippt::sqrt(uv.y));
	}

	static float3_t sqrt(float3_t uvw)
	{
		return make_float3(hippt::sqrt(uvw.x), hippt::sqrt(uvw.y), hippt::sqrt(uvw.z));
	}

	static float rsqrt(float x)
	{
		return 1.0f / hippt::sqrt(x);
	}

	static float length(float3_t u)
	{
		return hippt::sqrt(dot(u, u));
	}

	static float length(float2_t u)
	{
		return hippt::sqrt(dot(u, u));
	}

	static float length2(float3_t u)
	{
		return dot(u, u);
	}

	static float3_t abs(float3_t u)
	{
		return make_float3(std::abs(u.x), std::abs(u.y), std::abs(u.z));
	}

	template <typename T>
	static T abs(T a)
	{
		return std::abs(a);
	}

	static float fma(float a, float b, float c)
	{
		return a * b + c;
	}

	static float2_t fma(float2_t a, float2_t b, float2_t c)
	{
		return make_float2(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y));
	}

	static float3_t fma(float3_t a, float3_t b, float3_t c)
	{
		return make_float3(hippt::fma(a.x, b.x, c.x), hippt::fma(a.y, b.y, c.y), hippt::fma(a.z, b.z, c.z));
	}

	/*! An implementation of mix() using two fused-multiply add instructions. Used
	because the native mix() implementation had stability issues in a few
	spots. Credit to Fabian Giessen's blog, see:
	https://fgiesen.wordpress.com/2012/08/15/linear-interpolation-past-present-and-future/
	*/
	static float mix_fma(float x, float y, float a)
	{
		return hippt::fma(a, y, hippt::fma(-a, x, x));
	}

	template <typename T>
	static T max(T a, T b)
	{
		return a > b ? a : b;
	}

	/**
	 * Component-wise max of float3_t and int3_t
	 */
	template <>
	float3_t max(float3_t a, float3_t b)
	{
		return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
	}

	template <>
	int3_t max(int3_t a, int3_t b)
	{
		return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
	}

	template <typename T>
	static T min(T a, T b)
	{
		return a < b ? a : b;
	}

	/**
	 * Component-wise min of float3_t and int3_t
	 */
	template <>
	float3_t min(float3_t a, float3_t b)
	{
		return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
	}

	template <>
	int3_t min(int3_t a, int3_t b)
	{
		return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
	}

	/**
	 * Minimum of each component of the float3_t against x
	 */
	static float3_t min(float3_t a, float x)
	{
		return make_float3(min(a.x, x), min(a.y, x), min(a.z, x));
	}

	static float3_t min(float x, float3_t a)
	{
		return hippt::min(a, x);
	}

	/**
	 * Minimum of each component of the mat 2x2 against x
	 */
	static float2x2 min(float x, float2x2 a)
	{
		return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x));
	}

	static float2x2 min(float2x2 a, float x)
	{
		return float2x2(hippt::min(a.m[0][0], x), hippt::min(a.m[0][1], x), hippt::min(a.m[1][0], x), hippt::min(a.m[1][1], x));
	}

	template <typename T>
	static T clamp(T min_val, T max_val, T val)
	{
		return min(max_val, max(min_val, val));
	}

	static float2_t cos(float2_t x)
	{
		return make_float2(std::cos(x.x), std::cos(x.y));
	}

	static float3_t cos(float3_t x)
	{
		return make_float3(std::cos(x.x), std::cos(x.y), std::cos(x.z));
	}

	static float intrin_cosf(float x)
	{
		return std::cos(x);
	}

	static float3_t intrin_cosf(float3_t x)
	{
		return make_float3(std::cos(x.x), std::cos(x.y), std::cos(x.z));
	}

	static float2_t sin(float2_t x)
	{
		return make_float2(std::sin(x.x), std::sin(x.y));
	}

	static float3_t sin(float3_t x)
	{
		return make_float3(std::sin(x.x), std::sin(x.y), std::sin(x.z));
	}

	static float intrin_sinf(float x)
	{
		return std::sin(x);
	}

	static float intrin_expf(float x)
	{
		return expf(x);
	}

	static float3_t intrin_expf(float3_t x)
	{
		return make_float3(expf(x.x), expf(x.y), expf(x.z));
	}

	static float intrin_logf(float x)
	{
		return logf(x);
	}

	static float3_t atan2(float3_t y, float3_t x)
	{
		return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), atan2f(y.z, x.z));
	}

	static float2_t exp(float2_t x)
	{
		return make_float2(expf(x.x), expf(x.y));
	}

	static float3_t exp(float3_t x)
	{
		return make_float3(expf(x.x), expf(x.y), expf(x.z));
	}

	static float intrin_expm1f(float x)
	{
		return hippt::intrin_expf(x) - 1.0f;
	}

	static float3_t ldexp(float3_t x, int exp)
	{
		return make_float3(std::ldexp(x.x, exp), std::ldexp(x.y, exp), std::ldexp(x.z, exp));
	}

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
		constexpr float ERFC_SMALL		 = 0.0053854f;

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
	static T square(T x)
	{
		return x * x;
	}

	static float pow_1_4(float x)
	{
		return hippt::sqrt(hippt::sqrt(x));
	}

	static constexpr float pow_3(float x)
	{
		return x * x * x;
	}

	static constexpr float pow_4(float x)
	{
		float x2 = x * x;
		return x2 * x2;
	}

	static constexpr float pow_5(float x)
	{
		float x2 = x * x;
		float x4 = x2 * x2;
		return x4 * x;
	}

	static constexpr float pow_6(float x)
	{
		float x2 = x * x;
		float x4 = x2 * x2;
		return x4 * x2;
	}

	static float intrin_pow(float x, float y)
	{
		return powf(x, y);
	}

	static float pow_2_2_fit(float x)
	{
		return (exp2f(0.718151f * x) - 1.0f - 0.503456f * x) * 7.07342f;
	}

	static float2_t normalize(float2_t u)
	{
		return u / hippt::sqrt(hippt::dot(u, u));
	}

	static float3_t normalize(float3_t u)
	{
		return make_float3(u.x, u.y, u.z) / hippt::sqrt(hippt::dot(u, u));
	}

	template <typename T>
	static bool is_nan(const T& v)
	{
		return std::isnan(v);
	}

	template <typename T>
	static constexpr bool is_inf(const T& v)
	{
		return std::isinf(v);
	}

	static bool is_zero(float x)
	{
		return x < NEAR_ZERO && x > -NEAR_ZERO;
	}

	static bool is_finite(float x)
	{
		return std::isfinite(x);
	}

	static unsigned int float_as_uint(float float_num)
	{
		return std::bit_cast<unsigned int>(float_num);
	}

	static float uint_as_float(unsigned int uint_num)
	{
		return std::bit_cast<float>(uint_num);
	}

	static float fp16_bits_to_fp32(unsigned short int half_bits)
	{
		// Source: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion

		const unsigned int e = (half_bits & 0x7C00) >> 10;			 // exponent
		const unsigned int m = (half_bits & 0x03FF) << 13;			 // mantissa
		const unsigned int v = hippt::float_as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format

		return hippt::uint_as_float((half_bits & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
									((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
	}

	// IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
	static unsigned short int fp32_to_fp16_bits(const float x)
	{
		// Source: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion

		const unsigned int b = hippt::float_as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
		const unsigned int e = (b & 0x7F800000) >> 23;				 // exponent
		const unsigned int m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
		return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
			   ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
	}

	static unsigned short int half_as_ushort(fp16 half)
	{
		// fp16 is just fp32 on the CPU , so we can just reinterpret the bits as a float and then convert to fp16 bits
		return fp32_to_fp16_bits(half);
	}

	/**
	 * Reads the 32-bit or 64-bit word old located at the address 'address'
	 * in global or shared memory and stores 'value' to memory at the same address.
	 *
	 * These two operations are performed in one atomic transaction.
	 *
	 * The function returns old.
	 */
	template <typename T>
	T atomic_exchange(std::atomic<T>* address, T value)
	{
		return address->exchange(value);
	}

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
		while (prev_value < value && !address->compare_exchange_weak(prev_value, value))
		{
		}

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
		while (prev_value > value && !address->compare_exchange_weak(prev_value, value))
		{
		}

		return prev_value;
	}

	/**
	 * The function returns the value at 'address' because the increment
	 */
	template <typename T>
	T atomic_fetch_add(std::atomic<T>* atomic_address, T increment)
	{
		return atomic_address->fetch_add(increment);
	}

	/**
	 * This one is just an overload such that the code compiles on the CPU but this is meant to be used in kernels that will only ever run on the GPU. This is
	 * just to make the CPU compiler happy
	 */
	template <typename T>
	T atomic_fetch_add_gpu(T* atomic_address, T increment)
	{
		// Should not be used on the CPU

		Debug::debugbreak();
		return 0;
	}

	template <typename T>
	T atomic_load(std::atomic<T>* atomic_address)
	{
		return atomic_address->load();
	}

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

	template <typename T>
	T atomic_compare_exchange_gpu(T* address, T expected, T new_value)
	{
		// Should not be used on the CPU

		Debug::debugbreak();
		return 0;
	}

	/**
	 * For t=0, returns a
	 */
	template <typename T>
	static T lerp(T a, T b, float t)
	{
		return (1.0f - t) * a + t * b;
	}

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

	static float fract(float a)
	{
		return a - floorf(a);
	}

	template <typename T>
	static unsigned int popc(T bitmask)
	{
		return std::popcount(bitmask);
	}

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

	template <typename T>
	static T clz(T bitfield)
	{
		std::countl_zero(bitfield);
	}

	static bool warp_any(unsigned int thread_mask, bool predicate)
	{
		return predicate;
	}

	/**
	 * Returns a bit mask whose bits are set to 1 for threads that evaluated the predicate to true.
	 */
	static unsigned long long int warp_ballot(unsigned int thread_mask, bool predicate)
	{
		return predicate ? 1 : 0;
	}

	static unsigned long long int warp_activemask()
	{
		return 1;
	}

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
	static T warp_shfl(T var, int srcLane, int width = 1)
	{
		return var;
	}

	/**
	 *  T can be a 32-bit integer type, 64-bit integer type or a single precision or double precision floating point type.
	 *  The warp shuffle functions exchange values between threads within a warp.
	 *
	 * The optional width argument specifies subgroups, in which the warp can be divided to share the variables. It has to be a power of two smaller than or
	 * equal to warpSize. If it is smaller than warpSize, the warp is grouped into separate groups, that are each indexed from 0 to width as if it was its own
	 * entity, and only the lanes within that subgroup participate in the shuffle. The lane indices in the subgroup are given by laneIdx % width.
	 *
	 * The 64-bit unsigned integer mask argument specifies the lanes of the warp that will participate. Each participating thread must have its own bit set in
	 * its mask argument, and all active threads specified in any mask argument must execute the same call with the same mask, otherwise the result is
	 * undefined. The implementation includes a static assert to check that the program source uses the correct type for the mask.
	 */
	template <typename T>
	static T warp_shfl_sync(unsigned long long int mask, T var, int srcLane, int width = 1)
	{
		return var;
	}

	/**
	 * Copy from a lane with higher ID relative to caller
	 */
	template <typename T>
	static T warp_shfl_down(T var, int delta, int width = 1)
	{
		return var;
	}

	/**
	 * Copy from a lane with lower ID relative to caller
	 */
	template <typename T>
	static T warp_shfl_up(T var, int delta, int width = 1)
	{
		return var;
	}

	template <typename T>
	static T warp_reduce_max(unsigned long long int mask, T variable)
	{
		return variable;
	}

	static void syncwarp(unsigned int mask) {}

	/**
	 * Returns the index within its warp (not group) of the calling thread
	 *
	 * Warp sizes of 1 on the CPU
	 */
	static unsigned int warp_2D_thread_index()
	{
		return 1;
	}

	static void debugbreak()
	{
		Debug::debugbreak();
	}

	static float idx(float3_t v, int index)
	{
		return *(&v.x + index);
	}
#endif
} // namespace hippt

#ifndef __KERNELCC__

#include <iostream>
static std::ostream& operator<<(std::ostream& os, float3_t uvw)
{
	os << uvw.x << ", " << uvw.y << ", " << uvw.z;
	return os;
}

#endif

#endif
