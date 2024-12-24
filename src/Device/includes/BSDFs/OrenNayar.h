/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_OREN_NAYAR_H
#define DEVICE_OREN_NAYAR_H

#include "Device/includes/Sampling.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/Material/Material.h"

/* References:
 * [1] [Physically Based Rendering 3rd Edition] https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F oren_nayar_brdf_eval(const DeviceUnpackedEffectiveMaterial& material, const float3& local_view_direction, const float3& local_to_light_direction, float& pdf)
{
    // sin(theta)^2 = 1.0 - cos(theta)^2
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

        max_cos = hippt::max(0.0f, d_cos);
    }

    float sin_alpha, tan_beta;
    if (hippt::abs(local_to_light_direction.z) > hippt::abs(local_view_direction.z)) 
    {
        sin_alpha = sin_theta_o;
        tan_beta = sin_theta_i / hippt::abs(local_to_light_direction.z);
    }
    else 
    {
        sin_alpha = sin_theta_i;
        tan_beta = sin_theta_o / hippt::abs(local_view_direction.z);
    }

    float oren_nayar_A;
    float oren_nayar_B;
    MaterialUtils::get_oren_nayar_AB(material.oren_nayar_sigma, oren_nayar_A, oren_nayar_B);

    pdf = local_to_light_direction.z * M_INV_PI;
    return material.base_color * M_INV_PI * (oren_nayar_A + oren_nayar_B * max_cos * sin_alpha * tan_beta);
}

/**
 * Override of the eval function for world space directions
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F oren_nayar_brdf_eval(const DeviceUnpackedEffectiveMaterial& material, const float3& world_space_view_direction, const float3& surface_normal, const float3& world_space_to_light_direction, float& pdf)
{
    float3 T, B;
    build_ONB(surface_normal, T, B);

    // Using local view and light directions to simply following computations
    float3 local_view_direction = world_to_local_frame(T, B, surface_normal, world_space_view_direction);
    float3 local_to_light_direction = world_to_local_frame(T, B, surface_normal, world_space_to_light_direction);

    return oren_nayar_brdf_eval(material, local_view_direction, local_to_light_direction, pdf);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F oren_nayar_brdf_sample(const DeviceUnpackedEffectiveMaterial& material, const float3& world_space_view_direction, const float3& shading_normal, float3& out_sampled_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    out_sampled_direction = cosine_weighted_sample_around_normal(shading_normal, random_number_generator);

    return oren_nayar_brdf_eval(material, world_space_view_direction, shading_normal, out_sampled_direction, pdf);
}

#endif
