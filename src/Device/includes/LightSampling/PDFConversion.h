/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PDF_CONVERSION_H
#define DEVICE_INCLUDES_PDF_CONVERSION_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/Math.h"

 /**
  * Returns the cosine term of the given light source normal and the direction to the light source
  * 'minus_direction_to_light' must be the direction *towards* the light but *negated*, such that
  * dot(light_source_normal, minus_direction_to_light) > 0.0f (if the light isn't backfacing us)
  *
  * This function does the branching that allows backfacing lights or not
  */
HIPRT_INLINE HIPRT_DEVICE float compute_cosine_term_at_light_source(float3 light_source_normal, float3 minus_direction_to_light)
{
    // The cosine term is the dot product between the light source normal and the direction to the shading point

#if DirectLightSamplingAllowBackfacingLights == KERNEL_OPTION_TRUE
    // abs() to allow backfacing lights
    return hippt::abs(hippt::dot(light_source_normal, minus_direction_to_light));
#else
    // clamping to 0 to disallow backfacing lights
    return hippt::max(0.0f, hippt::dot(light_source_normal, minus_direction_to_light));
#endif
}

HIPRT_INLINE HIPRT_HOST_DEVICE float area_to_solid_angle_pdf(float area_pdf, float distance, float cos_theta_at_light_source)
{
    if (cos_theta_at_light_source < 1.0e-8f)
        return 0.0f;

    return area_pdf * hippt::square(distance) / cos_theta_at_light_source;
}

HIPRT_INLINE HIPRT_HOST_DEVICE float solid_angle_to_area_pdf(float solid_angle_pdf, float distance, float cos_theta_at_light_source)
{
    if (cos_theta_at_light_source < 1.0e-8f)
        return 0.0f;

    return solid_angle_pdf / hippt::square(distance) * cos_theta_at_light_source;
}

#endif
