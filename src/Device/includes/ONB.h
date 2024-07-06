/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ONB_H
#define DEVICE_ONB_H

#include "HostDeviceCommon/Math.h"

HIPRT_HOST_DEVICE HIPRT_INLINE void build_ONB(const float3& N, float3& T, float3& B)
{
    float3 up = hippt::abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = hippt::normalize(hippt::cross(up, N));
    B = hippt::cross(N, T);
}

/*
 * Rotation of the basis around the normal by 'basis_rotation' radians
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void build_rotated_ONB(const float3& N, float3& T, float3& B, float basis_rotation)
{
    float3 up = hippt::abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = hippt::normalize(hippt::cross(up, N));

    // Rodrigues' rotation
    T = T * cos(basis_rotation) + hippt::cross(N, T) * sin(basis_rotation) + N * hippt::dot(N, T) * (1.0f - cos(basis_rotation));
    B = hippt::cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 local_to_world_frame(const float3& N, const float3& V)
{
    float3 T, B;
    build_ONB(N, T, B);

    return hippt::normalize(V.x * T + V.y * B + V.z * N);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 local_to_world_frame(const float3& T, const float3& B, const float3& N, const float3& V)
{
    return hippt::normalize(V.x * T + V.y * B + V.z * N);
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 world_to_local_frame(const float3& N, const float3& V)
{
    float3 T, B;
    build_ONB(N, T, B);

    return hippt::normalize(make_float3(hippt::dot(V, T), hippt::dot(V, B), hippt::dot(V, N)));
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 world_to_local_frame(const float3& T, const float3& B, const float3& N, const float3& V)
{
    return hippt::normalize(make_float3(hippt::dot(V, T), hippt::dot(V, B), hippt::dot(V, N)));
}

#endif