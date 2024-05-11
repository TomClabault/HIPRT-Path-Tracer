/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_DISPATCHER_H
#define DEVICE_DISPATCHER_H

#include "Device/includes/Disney.h"
#include "Device/includes/RayPayload.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB brdf_dispatcher_eval(const RendererMaterial& material, RayPayload& ray_payload, const float3& view_direction, const float3& surface_normal, const float3& to_light_direction, float& pdf)
{
    return disney_eval(material, ray_payload, view_direction, surface_normal, to_light_direction, pdf);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB brdf_dispatcher_sample(const RendererMaterial& material, RayPayload& ray_payload, const float3& view_direction, const float3& surface_normal, const float3& geometric_normal, float3& bounce_direction, float& brdf_pdf, Xorshift32Generator& random_number_generator)
{
    return disney_sample(material, ray_payload, view_direction, surface_normal, geometric_normal, bounce_direction, brdf_pdf, random_number_generator);
}

#endif
