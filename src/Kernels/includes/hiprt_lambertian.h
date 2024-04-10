#ifndef HIPRT_LAMBERTIAN_H
#define HIPRT_LAMBERTIAN_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/hiprt_onb.h"

__device__ Color hiprt_lambertian_brdf(const RendererMaterial& material, const hiprtFloat3& to_light_direction, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal)
{
    return material.base_color * M_1_PI;
}

#endif