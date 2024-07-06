/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GLASS_H
#define DEVICE_GLASS_H

#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/Material.h"
#include "Device/includes/Sampling.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB smooth_glass_bsdf(const RendererMaterial& material, float3& out_bounce_direction, const float3& ray_direction, float3& surface_normal, float eta_i, float eta_t, float& pdf, Xorshift32Generator& random_generator)
{
    // Clamping here because the dot product can eventually returns values less
    // than -1 or greater than 1 because of precision errors in the vectors
    // (in previous calculations)
    float cos_theta_i = hippt::clamp(-1.0f, 1.0f, hippt::dot(surface_normal, -ray_direction));

    if (cos_theta_i < 0.0f)
    {
        // We're inside the surface, we're going to flip the eta and the normal for
        // the calculations that follow
        // Note that this also flips the normal for the caller of this function
        // since the normal is passed by reference. This is useful since the normal
        // will be used for offsetting the new ray origin for example
        cos_theta_i = -cos_theta_i;
        surface_normal = -surface_normal;

        float temp = eta_i;
        eta_i = eta_t;
        eta_t = temp;
    }

    // Computing the proportion of reflected light using fresnel equations
    // We're going to use the result to decide whether to refract or reflect the
    // ray
    float fresnel_reflect = fresnel_dielectric(cos_theta_i, eta_i, eta_t);
    if (random_generator() <= fresnel_reflect)
    {
        // Reflect the ray

        out_bounce_direction = reflect_ray(-ray_direction, surface_normal);
        pdf = fresnel_reflect;

        return ColorRGB(fresnel_reflect) / hippt::dot(surface_normal, out_bounce_direction);
    }
    else
    {
        // Refract the ray

        float3 refract_direction;
        bool can_refract = refract_ray(-ray_direction, surface_normal, refract_direction, eta_t / eta_i);
        if (!can_refract)
            // Shouldn't happen but can because of floating point imprecisions
            return ColorRGB(0.0f);

        out_bounce_direction = refract_direction;
        surface_normal = -surface_normal;
        pdf = 1.0f - fresnel_reflect;

        return ColorRGB(1.0f - fresnel_reflect) * material.base_color / hippt::dot(out_bounce_direction, surface_normal);
    }
}

#endif