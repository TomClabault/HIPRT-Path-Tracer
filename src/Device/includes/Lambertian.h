/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LAMBERTIAN_H
#define DEVICE_LAMBERTIAN_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F hiprt_lambertian_brdf(const RendererMaterial& material, const float3& to_light_direction, const float3& view_direction, const float3& surface_normal)
{
    return material.base_color / M_PI;
}

#endif