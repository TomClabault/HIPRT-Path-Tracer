/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OREN_NAYAR_H
#define OREN_NAYAR_H

#include "Kernels/includes/HIPRT_common.h"

/* References:
 * [1] [Physically Based Rendering 3rd Edition] https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
 */
__device__ Color oren_nayar_eval(const RendererMaterial& material, const float3& view_direction, const float3& surface_normal, const float3& to_light_direction)
{
    float3 T, B;
    build_ONB(surface_normal, T, B);

    // Using local view and light directions to simply following computations
    float3 local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    float3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);

    // sin(theta) = 1.0 - cos(theta)^2
	float sin_theta_i = sqrt(1.0f - local_to_light_direction.z * local_to_light_direction.z);
	float sin_theta_o = sqrt(1.0f - local_view_direction.z * local_view_direction.z);

    // max_cos here is going to be cos(phi_to_light - phi_view_direction)
    // but computed as cos(phi_light) * cos(phi_view) + sin(phi_light) * sin(phi_view)
    // according to cos(a - b) = cos(a) * cos(b) + sin(a) * sin(b)
    float max_cos = 0;
    if (sin_theta_i > 1.0e-4f && sin_theta_o > 1.0e-4f) 
    {
        float sin_phi_i = local_to_light_direction.y / sin_theta_i;
        float cos_phi_i = local_to_light_direction.x / sin_theta_i;

        float sin_phi_o = local_view_direction.y / sin_theta_o;
        float cos_phi_o = local_view_direction.x / sin_theta_o;

        float d_cos = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o;

        max_cos = RT_MAX(0.0f, d_cos);
    }

    float sin_alpha, tan_beta;
    if (abs(local_to_light_direction.z) > abs(local_view_direction.z)) 
    {
        sin_alpha = sin_theta_o;
        tan_beta = sin_theta_i / abs(local_to_light_direction.z);
    }
    else 
    {
        sin_alpha = sin_theta_i;
        tan_beta = sin_theta_o / abs(local_view_direction.z);
    }

    return material.base_color / M_PI * (material.oren_nayar_A + material.oren_nayar_B * max_cos * sin_alpha * tan_beta);
}

#endif