/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_GI_INITIAL_CANDIDATES_UTILS_H
#define DEVICE_INCLUDES_RESTIR_GI_INITIAL_CANDIDATES_UTILS_H
 
#include "Device/includes/PathTracing.h"

 /**
 * Returns true if the bounce was sampled successfully,
 * false otherwise (is the BSDF sample failed, if russian roulette killed the sample, ...)
 */
HIPRT_HOST_DEVICE bool restir_gi_compute_next_indirect_bounce(HIPRTRenderData& render_data, RayPayload& ray_payload, 
    ColorRGB32F& ray_throughput_to_visible_point, ColorRGB32F& ray_throughput_to_sample_point, HitInfo& closest_hit_info,
    float3 view_direction, hiprtRay& out_ray, MISBSDFRayReuse& mis_reuse, Xorshift32Generator& random_number_generator, BSDFIncidentLightInfo* incident_light_info = nullptr, float* out_bsdf_pdf = nullptr)
{
    ColorRGB32F bsdf_color;
    float3 bounce_direction;
    float bsdf_pdf;
    path_tracing_sample_next_indirect_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf, mis_reuse, random_number_generator, incident_light_info);

    if (out_bsdf_pdf != nullptr)
        *out_bsdf_pdf = bsdf_pdf;

    // Terminate ray if bad sampling
    if (bsdf_pdf <= 0.0f)
        return false;

    if (ray_payload.bounce > 0)
    {
        // With ReSTIR GI, we want the outgoing radiance from the second hit to the camera hit
        // This means that we're basically not taking the first hit into account and so we're not
        // updating the throughput (or the ray_color either, see the main loop) on the bounce 0

        ray_throughput_to_visible_point = path_tracing_update_ray_throughput(render_data, ray_payload, closest_hit_info, ray_throughput_to_visible_point, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator);
        if (ray_throughput_to_visible_point.is_black())
            // Special case to indicate killed by russian roulette
            return false;

        if (ray_payload.bounce > 1)
            // And updating the ray throughput but starting at bounce 1 so that this gives
            // us the throughput after the sample point which is going to be used to compute
            // the outgoing radiance towards the sample point
            ray_throughput_to_sample_point = path_tracing_update_ray_throughput(render_data, ray_payload, closest_hit_info, ray_throughput_to_sample_point, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator, false);
    }
    else
    {
        if (ray_payload.bounce >= render_data.render_settings.russian_roulette_min_depth && render_data.render_settings.use_russian_roulette)
            // Advancing the random number generation just to match non-ReSTIR GI path tracing in terms of randomness
            random_number_generator();
    }

    out_ray.origin = closest_hit_info.inter_point;
    out_ray.direction = bounce_direction;

    return true;
}

#endif
