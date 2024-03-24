#ifndef HIPRT_ONB_H
#define HIPRT_ONB_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"

__device__ void buildONB(const hiprtFloat3& N, hiprtFloat3& T, hiprtFloat3& B)
{
    hiprtFloat3 up = abs(N.z) < 0.9999999 ? hiprtFloat3(0, 0, 1) : hiprtFloat3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
__device__ hiprtFloat3 local_to_world_frame(const hiprtFloat3& N, const hiprtFloat3& V)
{
    hiprtFloat3 T, B;
    buildONB(N, T, B);

    return V.x * T + V.y * B + V.z * N;
}

__device__ hiprtFloat3 local_to_world_frame(const hiprtFloat3& T, const hiprtFloat3& B, const hiprtFloat3& N, const hiprtFloat3& V)
{
    return V.x * T + V.y * B + V.z * N;
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
__device__ hiprtFloat3 world_to_local_frame(const hiprtFloat3& N, const hiprtFloat3& V)
{
    hiprtFloat3 T, B;
    buildONB(N, T, B);

    return hiprtFloat3(dot(V, T), dot(V, B), dot(V, N));
}

__device__ hiprtFloat3 world_to_local_frame(const hiprtFloat3& T, const hiprtFloat3& B, const hiprtFloat3& N, const hiprtFloat3& V)
{
    return hiprtFloat3(dot(V, T), dot(V, B), dot(V, N));
}

#endif