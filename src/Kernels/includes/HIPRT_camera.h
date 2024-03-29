#ifndef HIPRT_CAMERA_H
#define HIPRT_CAMERA_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"

struct HIPRTCamera
{
    float4x4 inverse_view;
    float4x4 inverse_projection;
    hiprtFloat3 position;

    __device__ __host__ hiprtRay get_camera_ray(float x, float y, int2 res)
    {
        float x_ndc_space = x / res.x * 2 - 1;
        float y_ndc_space = y / res.y * 2 - 1;

        hiprtFloat3 ray_origin_view_space = { 0.0f, 0.0f, 0.0f };
        hiprtFloat3 ray_origin = matrix_X_point(inverse_view, ray_origin_view_space);

        // Point on the near plane
        hiprtFloat3 ray_point_dir_ndc_homog = { x_ndc_space, y_ndc_space, -1.0f };
        hiprtFloat3 ray_point_dir_vs_homog = matrix_X_point(inverse_projection, ray_point_dir_ndc_homog);
        hiprtFloat3 ray_point_dir_vs = ray_point_dir_vs_homog;
        hiprtFloat3 ray_point_dir_ws = matrix_X_point(inverse_view, ray_point_dir_vs);

        hiprtFloat3 ray_direction = normalize(ray_point_dir_ws - ray_origin);

        hiprtRay ray;
        ray.origin = ray_origin;
        ray.direction = ray_direction;

        return ray;
    }
};

#endif