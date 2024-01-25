#pragma once

#include <hiprt/hiprt_common.h>

#if !defined( __KERNELCC__ )
struct hiprtInt2
{
	int x, y;
};

struct hiprtFloat2
{
	float x, y;
};

struct hiprtInt3
{
	int x, y, z;
};

struct hiprtFloat3
{
	float x, y, z;
};

struct hiprtInt4
{
	int x, y, z, w;
};

struct hiprtFloat4
{
	float x, y, z, w;
};

struct hiprtUint2
{
	unsigned int x, y;
};

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInt2 make_hiprtInt2( int x, int y ) { return { x, y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat2 make_hiprtFloat2( float x, float y ) { return { x, y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInt3 make_hiprtInt3( int x, int y, int z ) { return { x, y, z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat3 make_hiprtFloat3( float x, float y, float z ) { return { x, y, z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInt4 make_hiprtInt4( int x, int y, int z, int w ) { return { x, y, z, w }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtFloat4 make_hiprtFloat4( float x, float y, float z, float w ) { return { x, y, z, w }; }

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtUint2 make_hiprtUint2( unsigned int x, unsigned int y ) { return { x, y }; }

#if defined( HIPRT_EXPORTS )
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
#endif
#else
#define hiprtInt2 int2
#define hiprtInt3 int3
#define hiprtInt4 int4
#define hiprtUint2 uint2

#define hiprtFloat2 float2
#define hiprtFloat3 float3
#define hiprtFloat4 float4

#define make_hiprtInt2 make_int2
#define make_hiprtInt3 make_int3
#define make_hiprtInt4 make_int4
#define make_hiprtUint2 make_uint2

#define make_hiprtFloat2 make_float2
#define make_hiprtFloat3 make_float3
#define make_hiprtFloat4 make_float4
#endif
