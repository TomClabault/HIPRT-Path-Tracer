#ifndef HIPRT_ONB_H
#define HIPRT_ONB_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"

__device__ void build_ONB(const hiprtFloat3& N, hiprtFloat3& T, hiprtFloat3& B)
{
    hiprtFloat3 up = abs(N.z) < 0.9999999 ? hiprtFloat3(0, 0, 1) : hiprtFloat3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

/*
 * Rotation of the basis around the normal by 'basis_rotation' radians
 */
__device__ void build_rotated_ONB(const hiprtFloat3& N, hiprtFloat3& T, hiprtFloat3& B, float basis_rotation)
{
    hiprtFloat3 up = abs(N.z) < 0.9999999 ? hiprtFloat3(0, 0, 1) : hiprtFloat3(1, 0, 0);
    T = normalize(cross(up, N));

    // Rodrigues' rotation
    T = T * cos(basis_rotation) + cross(N, T) * sin(basis_rotation) + N * dot(N, T) * (1.0f - cos(basis_rotation));
    B = cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
__device__ hiprtFloat3 local_to_world_frame(const hiprtFloat3& N, const hiprtFloat3& V)
{
    hiprtFloat3 T, B;
    build_ONB(N, T, B);

    return normalize(V.x * T + V.y * B + V.z * N);
}

__device__ hiprtFloat3 local_to_world_frame(const hiprtFloat3& T, const hiprtFloat3& B, const hiprtFloat3& N, const hiprtFloat3& V)
{
    return normalize(V.x * T + V.y * B + V.z * N);
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
__device__ hiprtFloat3 world_to_local_frame(const hiprtFloat3& N, const hiprtFloat3& V)
{
    hiprtFloat3 T, B;
    build_ONB(N, T, B);

    return normalize(hiprtFloat3(dot(V, T), dot(V, B), dot(V, N)));
}

__device__ hiprtFloat3 world_to_local_frame(const hiprtFloat3& T, const hiprtFloat3& B, const hiprtFloat3& N, const hiprtFloat3& V)
{
    return normalize(hiprtFloat3(dot(V, T), dot(V, B), dot(V, N)));
}

#endif