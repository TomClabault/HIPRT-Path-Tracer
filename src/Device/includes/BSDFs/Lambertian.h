/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LAMBERTIAN_H
#define DEVICE_LAMBERTIAN_H

#include "Device/includes/ONB.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F lambertian_brdf_eval(const DeviceUnpackedEffectiveMaterial& material, float NoL, float& pdf)
{
    pdf = 0.0f;

    if (NoL <= 0.0f)
        return ColorRGB32F(0.0f);

    pdf = NoL * M_INV_PI;
    return material.base_color * M_INV_PI;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F lambertian_brdf_sample(
    const DeviceUnpackedEffectiveMaterial& material, 
    const float3& shading_normal, float3& sampled_direction, 
    float& pdf, Xorshift32Generator& random_number_generator, BSDFIncidentLightInfo& out_sampled_light_info)
{
    sampled_direction = cosine_weighted_sample_around_normal_world_space(shading_normal, random_number_generator);
    
    out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;

    return lambertian_brdf_eval(material, hippt::dot(shading_normal, sampled_direction), pdf);
}

#endif
