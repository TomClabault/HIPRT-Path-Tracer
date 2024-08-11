/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_HIPRT_CAMERA_H
#define HOST_DEVICE_COMMON_HIPRT_CAMERA_H

#include "HostDeviceCommon/Math.h"

#include <hiprt/hiprt_types.h> // for hiprtRay

/**
 * Simplified camera class passed to the shader
 */
struct HIPRTCamera
{
    float4x4 inverse_view;
    float4x4 inverse_projection;

    HIPRT_HOST_DEVICE hiprtRay get_camera_ray(float x, float y, int2 res)
    {
        float x_ndc_space = x / res.x * 2 - 1;
        float y_ndc_space = y / res.y * 2 - 1;

        float3 ray_origin_view_space = { 0.0f, 0.0f, 0.0f };
        float3 ray_origin = matrix_X_point(inverse_view, ray_origin_view_space);

        // Point on the near plane
        float3 ray_point_dir_ndc_homog = { x_ndc_space, y_ndc_space, -1.0f };
        float3 ray_point_dir_vs_homog = matrix_X_point(inverse_projection, ray_point_dir_ndc_homog);
        float3 ray_point_dir_vs = ray_point_dir_vs_homog;
        float3 ray_point_dir_ws = matrix_X_point(inverse_view, ray_point_dir_vs);

        float3 ray_direction = hippt::normalize(ray_point_dir_ws - ray_origin);

        hiprtRay ray;
        ray.origin = ray_origin;
        ray.direction = ray_direction;

        return ray;
    }
};

#endif