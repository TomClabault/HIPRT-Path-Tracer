/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_ONB_H
#define HIPRT_ONB_H

#include "HostDeviceCommon/math.h"

__device__ void build_ONB(const float3& N, float3& T, float3& B)
{
    float3 up = abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = hiprtpt::normalize(hiprtpt::cross(up, N));
    B = hiprtpt::cross(N, T);
}

/*
 * Rotation of the basis around the normal by 'basis_rotation' radians
 */
__device__ void build_rotated_ONB(const float3& N, float3& T, float3& B, float basis_rotation)
{
    float3 up = abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = hiprtpt::normalize(hiprtpt::cross(up, N));

    // Rodrigues' rotation
    T = T * cos(basis_rotation) + hiprtpt::cross(N, T) * sin(basis_rotation) + N * hiprtpt::dot(N, T) * (1.0f - cos(basis_rotation));
    B = hiprtpt::cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
__device__ float3 local_to_world_frame(const float3& N, const float3& V)
{
    float3 T, B;
    build_ONB(N, T, B);

    return hiprtpt::normalize(V.x * T + V.y * B + V.z * N);
}

__device__ float3 local_to_world_frame(const float3& T, const float3& B, const float3& N, const float3& V)
{
    return hiprtpt::normalize(V.x * T + V.y * B + V.z * N);
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
__device__ float3 world_to_local_frame(const float3& N, const float3& V)
{
    float3 T, B;
    build_ONB(N, T, B);

    return hiprtpt::normalize(make_float3(hiprtpt::dot(V, T), hiprtpt::dot(V, B), hiprtpt::dot(V, N)));
}

__device__ float3 world_to_local_frame(const float3& T, const float3& B, const float3& N, const float3& V)
{
    return hiprtpt::normalize(make_float3(hiprtpt::dot(V, T), hiprtpt::dot(V, B), hiprtpt::dot(V, N)));
}

#endif