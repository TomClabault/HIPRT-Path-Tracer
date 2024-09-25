/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LAMBERTIAN_H
#define DEVICE_LAMBERTIAN_H

#include "Device/includes/ONB.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F lambertian_brdf_eval(const SimplifiedRendererMaterial& material, const float3& view_direction, const float3& surface_normal, const float3& to_light_direction, float& pdf)
{
    pdf = 0.0f;

    float cos_theta = hippt::dot(to_light_direction, surface_normal);
    if (cos_theta <= 0.0f)
        return ColorRGB32F(0.0f);

    pdf = cos_theta / M_PI;
    return material.base_color / M_PI;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F lambertian_brdf_sample(const SimplifiedRendererMaterial& material, const float3& view_direction, const float3& shading_normal, float3& sampled_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    sampled_direction = cosine_weighted_sample(shading_normal, random_number_generator);

    return lambertian_brdf_eval(material, view_direction, shading_normal, sampled_direction, pdf);
}

#endif
