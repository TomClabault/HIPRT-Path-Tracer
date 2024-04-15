/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_LAMBERTIAN_H
#define HIPRT_LAMBERTIAN_H

#include "Device/includes/onb.h"

__device__ Color hiprt_lambertian_brdf(const RendererMaterial& material, const float3& to_light_direction, const float3& view_direction, const float3& surface_normal)
{
    return material.base_color * M_1_PI;
}

#endif