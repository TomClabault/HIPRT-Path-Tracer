/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_ONB_H
#define HIPRT_ONB_H

#include "Kernels/includes/HIPRT_common.h"

__device__ void build_ONB(const float3& N, float3& T, float3& B)
{
    float3 up = abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

/*
 * Rotation of the basis around the normal by 'basis_rotation' radians
 */
__device__ void build_rotated_ONB(const float3& N, float3& T, float3& B, float basis_rotation)
{
    float3 up = abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = normalize(cross(up, N));

    // Rodrigues' rotation
    T = T * cos(basis_rotation) + cross(N, T) * sin(basis_rotation) + N * dot(N, T) * (1.0f - cos(basis_rotation));
    B = cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
__device__ float3 local_to_world_frame(const float3& N, const float3& V)
{
    float3 T, B;
    build_ONB(N, T, B);

    return normalize(V.x * T + V.y * B + V.z * N);
}

__device__ float3 local_to_world_frame(const float3& T, const float3& B, const float3& N, const float3& V)
{
    return normalize(V.x * T + V.y * B + V.z * N);
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
__device__ float3 world_to_local_frame(const float3& N, const float3& V)
{
    float3 T, B;
    build_ONB(N, T, B);

    return normalize(make_float3(dot(V, T), dot(V, B), dot(V, N)));
}

__device__ float3 world_to_local_frame(const float3& T, const float3& B, const float3& N, const float3& V)
{
    return normalize(make_float3(dot(V, T), dot(V, B), dot(V, N)));
}

#endif