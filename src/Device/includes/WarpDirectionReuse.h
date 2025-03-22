/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WARP_DIRECTION_REUSE_H
#define DEVICE_INCLUDES_WARP_DIRECTION_REUSE_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"

#include "HostDeviceCommon/HitInfo.h"

/**
 * Experimental implementation of [Generate Coherent Rays Directly, Liu et al., 2024]
 * 
 * This incomplete implementation supposes that all threads in the warp have the same material
 * type and this does not implement the "interleaved groups" approach to reduce correlation
 * 
 * Preliminary results only show a 10% boost in perf, even without correlation reduction and on
 * the Bistro (which is an expensive scene to trace). Because the correlations were pretty bad,
 * the implementation of the paper was discontinued
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void warp_direction_reuse(const HIPRTRenderData& render_data, const HitInfo& closest_hit_info, RayPayload& ray_payload, float3 view_direction, float3& in_out_bounce_direction, ColorRGB32F& out_bsdf_color, float& out_bsdf_pdf, int bounce, Xorshift32Generator& random_number_generator)
{
    if (bounce == 0)
    {
        // Direction reuse is only done on the first bounce because the efficiency largely decreases at later bounces
        
        unsigned int active_mask = hippt::warp_activemask();
        unsigned int first_active_thread_index = hippt::ffs(active_mask) - 1;

        float3 local_direction = world_to_local_frame(closest_hit_info.shading_normal, in_out_bounce_direction);

        local_direction.x = hippt::warp_shfl(local_direction.x, first_active_thread_index);
        local_direction.y = hippt::warp_shfl(local_direction.y, first_active_thread_index);
        local_direction.z = hippt::warp_shfl(local_direction.z, first_active_thread_index);

        in_out_bounce_direction = local_to_world_frame(closest_hit_info.shading_normal, local_direction);

        BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
        BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, in_out_bounce_direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness);
        out_bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, out_bsdf_pdf, random_number_generator);
    }
}

#endif
