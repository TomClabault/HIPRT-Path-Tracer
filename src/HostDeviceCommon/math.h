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

#define RT_MIN( a, b ) ( ( ( b ) < ( a ) ) ? ( b ) : ( a ) )
#define RT_MAX( a, b ) ( ( ( b ) > ( a ) ) ? ( b ) : ( a ) )

#ifdef __KERNELCC__
#define normalize hiprt::normalize
#define cross hiprt::cross
#define dot hiprt::dot

#define M_PI hiprt::Pi
#else
#define normalize normalize
#define cross cross
#define dot dot
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
