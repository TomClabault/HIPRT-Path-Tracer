/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_GI_INITIAL_CANDIDATES_H

#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/Lights.h"
#include "Device/includes/LightSampling/LightUtils.h"
#include "Device/includes/ReSTIR/GI/InitialCandidatesUtils.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/TargetFunction.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_InitialCandidates(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_InitialCandidates(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    if (!render_data.aux_buffers.pixel_active[pixel_index])
        // Pixel isn't active because of adaptive sampling or render resolution scaling
        return;

    if (render_data.render_settings.do_render_low_resolution())
        // Reducing the number of bounces to 3 if rendering at low resolution
        // for better interactivity
        render_data.render_settings.nb_bounces = hippt::min(3, render_data.render_settings.nb_bounces);



    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator random_number_generator(seed);

    // Initializing the closest hit info the information from the camera ray pass
    HitInfo closest_hit_info;
    closest_hit_info.inter_point = render_data.g_buffer.primary_hit_position[pixel_index];
    closest_hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
    closest_hit_info.shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
    closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];

    // Initializing the ray with the information from the camera ray pass
    hiprtRay ray;
    ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    RayPayload ray_payload;
    ray_payload.next_ray_state = RayState::BOUNCE;
    ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();

    // Because this is the camera hit (and assuming the camera isn't inside volumes for now),
    // the ray volume state after the camera hit is just an empty interior stack but with
    // the material index that we hit pushed onto the stack. That's it. Because it is that
    // simple, we don't have the ray volume state in the GBuffer but rather we can
    // reconstruct the ray volume state on the fly
    ray_payload.volume_state.reconstruct_first_hit(
        ray_payload.material,
        render_data.buffers.material_indices,
        closest_hit_info.primitive_index,
        random_number_generator);

    // This structure is going to contain the information for reusing the
    // BSDF ray when doing NEE with MIS: as a matter of fact, when doing
    // NEE with MIS, we're shooting a BSDF ray. If that ray doesn't hit
    // an emissive triangle, we can just reuse that ray for our indirect
    // bounce ray. That structure contains all that is necessary to reuse
    // the ray. This structure is filled by the emissive light sampling
    // or the envmap sampling function
    MISBSDFRayReuse mis_reuse;
    bool intersection_found = closest_hit_info.primitive_index != -1;

    ReSTIRSurface initial_surface;
    initial_surface.geometric_normal = closest_hit_info.geometric_normal;
    initial_surface.shading_normal = closest_hit_info.shading_normal;
    initial_surface.primitive_index = closest_hit_info.primitive_index;
    initial_surface.material = ray_payload.material;
    initial_surface.ray_volume_state = ray_payload.volume_state;
    initial_surface.shading_point = closest_hit_info.inter_point;
    initial_surface.view_direction = -ray.direction;

    float bsdf_sample_pdf = 0.0f;
    ReSTIRGISample restir_gi_initial_sample;

    ColorRGB32F incoming_radiance_to_visible_point;
    ColorRGB32F incoming_radiance_to_sample_point;
    ColorRGB32F throughput_to_visible_point = ColorRGB32F(1.0f);

    // + 1 to nb_bounces here because we want "0" bounces to still act as one
    // hit and to return some color
    for (int& bounce = ray_payload.bounce; bounce < render_data.render_settings.nb_bounces + 1; bounce++)
    {
        if (ray_payload.next_ray_state != RayState::MISSED)
        {
            if (bounce > 0)
            {
                if (bounce == 1)
                    // This is going to be tracing the ray from the visible point to the sample:
                    // we're saving the random seed used during the BVH traversal to be able to reproduce
                    // alpha tests
                    restir_gi_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;

                intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, mis_reuse, random_number_generator);
            }

            if (intersection_found)
            {
                if (bounce == 0)
                    store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);
                else if (bounce > 0)
                    // Storing data for ReGIR representative points
                    ReGIR_update_representative_data(render_data, closest_hit_info.inter_point, closest_hit_info.shading_normal, render_data.current_camera, closest_hit_info.primitive_index, false, ray_payload.material);

                if (bounce == 1)
                {
                    restir_gi_initial_sample.sample_point_geometric_normal.pack(closest_hit_info.geometric_normal);
                    restir_gi_initial_sample.sample_point = closest_hit_info.inter_point;
                    restir_gi_initial_sample.sample_point_primitive_index = closest_hit_info.primitive_index;
                    restir_gi_initial_sample.sample_point_rough_enough = MaterialUtils::can_do_light_sampling(ray_payload.material, render_data.render_settings.restir_gi_settings.neighbor_sample_point_roughness_threshold);
                }

                if (bounce > 0)
                {
                    // Estimating with a throughput of 1.0f here because we're going to apply the throughput ourselves
                    ColorRGB32F direct_lighting_estimation = estimate_direct_lighting(render_data, ray_payload, ColorRGB32F(1.0f), closest_hit_info, -ray.direction, x, y, mis_reuse, random_number_generator);
                    // Updating the cumulated outgoing radiance of our path to the visible point
                    incoming_radiance_to_visible_point += clamp_direct_lighting_estimation(direct_lighting_estimation * throughput_to_visible_point, render_data.render_settings.indirect_contribution_clamp, bounce);
                }

                float bsdf_pdf;
                BSDFIncidentLightInfo incident_light_info;
                bool valid_indirect_bounce = restir_gi_compute_next_indirect_bounce(render_data, ray_payload, throughput_to_visible_point, closest_hit_info, -ray.direction, ray, mis_reuse, random_number_generator, &incident_light_info, &bsdf_pdf);
                if (!valid_indirect_bounce)
                    // Bad BSDF sample (under the surface), killed by russian roulette, ...
                    break;

                if (bounce == 0)
                {
                    restir_gi_initial_sample.incident_light_info_at_visible_point = incident_light_info;
                    bsdf_sample_pdf = bsdf_pdf;
                }
            }
            else
            {
                if (bounce == 1)
                {
                    // For envmap path, the direction is stored in the hit point
                    restir_gi_initial_sample.sample_point = ray.direction;
                    // -1 for the primitive index indicates that this is an envmap sample
                    restir_gi_initial_sample.sample_point_primitive_index = -1;
                }

                incoming_radiance_to_visible_point += path_tracing_miss_gather_envmap(render_data, throughput_to_visible_point, ray.direction, ray_payload.bounce, pixel_index);

                ray_payload.next_ray_state = RayState::MISSED;
            }
        }
        else if (ray_payload.next_ray_state == RayState::MISSED)
            break;
    }

    // Checking for NaNs / negative value samples. Output 
    if (!sanity_check(render_data, ray_payload.ray_color, x, y))
        return;

    // If we got here, this means that we still have at least one ray active
    // This is a concurrent write by the way but we don't really care, everyone is writing
    // the same value
    render_data.aux_buffers.still_one_ray_active[0] = 1;

    restir_gi_initial_sample.incoming_radiance_to_visible_point.pack(incoming_radiance_to_visible_point);
    restir_gi_initial_sample.target_function = ReSTIR_GI_evaluate_target_function<true, false>(render_data, restir_gi_initial_sample, initial_surface, random_number_generator);

    float resampling_weight = 0.0f;
    float mis_weight = 1.0f;
    float target_function = restir_gi_initial_sample.target_function;
    float source_pdf = bsdf_sample_pdf;
    if (source_pdf > 0.0f)
        resampling_weight = mis_weight * restir_gi_initial_sample.target_function / source_pdf;

    ReSTIRGIReservoir restir_gi_initial_reservoir;
    restir_gi_initial_reservoir.add_one_candidate(restir_gi_initial_sample, resampling_weight, random_number_generator);
    restir_gi_initial_reservoir.end();
    restir_gi_initial_reservoir.sanity_check(make_int2(x, y));

    render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer[pixel_index] = restir_gi_initial_reservoir;
}

#endif
