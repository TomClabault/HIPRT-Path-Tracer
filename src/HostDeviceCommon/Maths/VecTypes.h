/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_VEC_TYPES_H
#define HOST_DEVICE_COMMON_VEC_TYPES_H

#include <hiprt/hiprt_vec.h>

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
