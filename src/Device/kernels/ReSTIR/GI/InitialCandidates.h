/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_GI_INITIAL_CANDIDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/RussianRoulette.h"
#include "Device/includes/SanityCheck.h"
#include "Device/includes/WarpDirectionReuse.h"

#include "HostDeviceCommon/Xorshift.h"

#ifndef __KERNELCC__
#include "Utils/Utils.h" // For debugbreak in sanity_check()

// For logging stuff on the CPU and avoid everything being mixed
// up in the terminal because of multithreading
#include <mutex>
std::mutex g_mutex;
#endif

HIPRT_HOST_DEVICE void store_denoiser_AOVs(HIPRTRenderData& render_data, uint32_t pixel_index, float3 shading_normal, ColorRGB32F base_color)
{
    if (render_data.render_settings.sample_number == 0)
        render_data.aux_buffers.denoiser_albedo[pixel_index] = base_color;
    else
        render_data.aux_buffers.denoiser_albedo[pixel_index] = (render_data.aux_buffers.denoiser_albedo[pixel_index] * render_data.render_settings.denoiser_AOV_accumulation_counter + base_color) / (render_data.render_settings.denoiser_AOV_accumulation_counter + 1.0f);

    if (render_data.render_settings.sample_number == 0)
        render_data.aux_buffers.denoiser_normals[pixel_index] = shading_normal;
    else
    {
        float3 accumulated_normal = (render_data.aux_buffers.denoiser_normals[pixel_index] * render_data.render_settings.denoiser_AOV_accumulation_counter + shading_normal) / (render_data.render_settings.denoiser_AOV_accumulation_counter + 1.0f);
        float normal_length = hippt::length(accumulated_normal);
        if (!hippt::is_zero(normal_length))
            // Checking that it is non-zero otherwise we would accumulate a persistent NaN in the buffer when normalizing by the 0-length
            render_data.aux_buffers.denoiser_normals[pixel_index] = accumulated_normal / normal_length;
    }
}

HIPRT_HOST_DEVICE void output_ReSTIR_GI_initial_candidate(HIPRTRenderData& render_data, uint32_t pixel_index)
{
    ReSTIRGIReservoir initial_candidate;
}

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
        return;

    if (render_data.render_settings.do_render_low_resolution())
        // Reducing the number of bounces to 3 if rendering at low resolution
        // for better interactivity
        render_data.render_settings.nb_bounces = hippt::min(3, render_data.render_settings.nb_bounces);

#if ViewportColorOverriden == 1
    // If some kernel option is going to debug some color in the viewport,
    // then we're clearing the viewport buffer here
    render_data.buffers.accumulated_ray_colors[pixel_index] = ColorRGB32F();
#endif


    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);
    Xorshift32Generator random_number_generator(seed);

    // Initializing the closest hit info the information from the camera ray pass
    HitInfo closest_hit_info;
    closest_hit_info.inter_point = render_data.g_buffer.primary_hit_position[pixel_index];
    closest_hit_info.geometric_normal = hippt::normalize(render_data.g_buffer.geometric_normals[pixel_index].unpack());
    closest_hit_info.shading_normal = hippt::normalize(render_data.g_buffer.shading_normals[pixel_index].unpack());
    closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];

    // Initializing the ray with the information from the camera ray pass
    hiprtRay ray;
    ray.direction = hippt::normalize(-render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index));

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

    ReSTIRGISample restir_gi_initial_candidate;

    // + 1 to nb_bounces here because we want "0" bounces to still act as one
    // hit and to return some color
    for (int& bounce = ray_payload.bounce; bounce < render_data.render_settings.nb_bounces + 1; bounce++)
    {
        if (ray_payload.next_ray_state != RayState::MISSED)
        {
            if (bounce > 0)
                intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, bounce, mis_reuse, random_number_generator);

            if (intersection_found)
            {
                if (bounce == 0)
                    store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);

                // For the BRDF calculations, bounces, ... to be correct, we need the normal to be in the same hemisphere as
                // the view direction. One thing that can go wrong is when we have an emissive triangle (typical area light)
                // and a ray hits the back of the triangle. The normal will not be facing the view direction in this
                // case and this will cause issues later in the BRDF.
                // Because we want to allow backfacing emissive geometry (making the emissive geometry double sided
                // and emitting light in both directions of the surface), we're negating the normal to make
                // it face the view direction (but only for emissive geometry)
                if (ray_payload.material.is_emissive() && hippt::dot(-ray.direction, closest_hit_info.geometric_normal) < 0)
                {
                    closest_hit_info.geometric_normal = -closest_hit_info.geometric_normal;
                    closest_hit_info.shading_normal = -closest_hit_info.shading_normal;
                }

                // --------------------------------------------------- //
                // ----------------- Direct lighting ----------------- //
                // --------------------------------------------------- //

                // Estimates direct lighting with next-event estimation and directly modifies ray_payload.ray_color
                estimate_direct_lighting(render_data, ray_payload, closest_hit_info, -ray.direction, x, y, mis_reuse, random_number_generator);

                // --------------------------------------- //
                // ---------- Indirect lighting ---------- //
                // --------------------------------------- //

                float bsdf_pdf;
                float3 bounce_direction;
                ColorRGB32F bsdf_color;

                if (mis_reuse.has_ray())
                    bsdf_color = reuse_mis_bsdf_sample(bounce_direction, bsdf_pdf, ray_payload, mis_reuse);
                else
                    bsdf_color = bsdf_dispatcher_sample(render_data, ray_payload.material, ray_payload.volume_state, true, 
                                                        -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bounce_direction, 
                                                        bsdf_pdf, random_number_generator, bounce);
#if DoFirstBounceWarpDirectionReuse
                warp_direction_reuse(render_data, closest_hit_info, ray_payload, -ray.direction, bounce_direction, bsdf_color, bsdf_pdf, bounce, random_number_generator);
#endif

                // Terminate ray if bad sampling
                if (bsdf_pdf <= 0.0f)
                    break;

                ColorRGB32F throughput_attenuation = bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal)) / bsdf_pdf;
                // Russian roulette
                if (!do_russian_roulette(render_data.render_settings, bounce, ray_payload.throughput, throughput_attenuation, random_number_generator))
                    break;

                // Dispersion ray throughput filter
                ray_payload.throughput *= get_dispersion_ray_color(ray_payload.volume_state.sampled_wavelength, ray_payload.material.dispersion_scale);
                ray_payload.throughput *= throughput_attenuation;
                ray_payload.next_ray_state = RayState::BOUNCE;

                ray.origin = closest_hit_info.inter_point;
                ray.direction = bounce_direction;
            }
            else
            {
                ColorRGB32F skysphere_color;

                if (render_data.world_settings.ambient_light_type == AmbientLightType::UNIFORM || render_data.bsdfs_data.white_furnace_mode)
                    skysphere_color = render_data.world_settings.uniform_light_color;
                else if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
                {
#if EnvmapSamplingStrategy != ESS_NO_SAMPLING
                    // If we have sampling, only taking envmap into account on camera ray miss
                    if (bounce == 0)
#endif
                    {
                        // We're only getting the skysphere radiance for the first rays because the
                        // syksphere is importance sampled.
                        skysphere_color = eval_envmap_no_pdf(render_data.world_settings, ray.direction);

#if EnvmapSamplingStrategy == ESS_NO_SAMPLING
                        // If we don't have envmap sampling, we're only going to unscale on
                        // bounce 0 (which is when a ray misses directly --> background color).
                        // Otherwise, if not bounce 2, we do want to take the scaling into
                        // account so this if will fail and the envmap color will never be unscaled
                        if (!render_data.world_settings.envmap_scale_background_intensity && bounce == 0)
#else
                        if (!render_data.world_settings.envmap_scale_background_intensity)
#endif
                            // Un-scaling the envmap if the user doesn't want to scale the background
                            skysphere_color /= render_data.world_settings.envmap_intensity;
                    }
                }

                skysphere_color = clamp_light_contribution(skysphere_color, render_data.render_settings.envmap_contribution_clamp, /* clamp condition */ true);

                ColorRGB32F indirect_lighting_contribution = skysphere_color * ray_payload.throughput;
                // Only clamping with the indirect lighting clamp value if
                // this is bounce > 0 (thanks to /* clamp condition */ bounce > 0)
                ColorRGB32F clamped_indirect_lighting_contribution = clamp_light_contribution(
                    indirect_lighting_contribution, render_data.render_settings.indirect_contribution_clamp, 
                    /* clamp condition */ bounce > 0);

                ray_payload.ray_color += clamped_indirect_lighting_contribution;
                ray_payload.next_ray_state = RayState::MISSED;

                if (bounce == 0)
                    // The camera ray missed so we don't have the normals but we have the base color
                    store_denoiser_AOVs(render_data, pixel_index, make_float3(0, 0, 0), skysphere_color);
            }
        }
        else if (ray_payload.next_ray_state == RayState::MISSED)
            break;
    }

    // Checking for NaNs / negative value samples. Output 
    if (!sanity_check(render_data, ray_payload, x, y))
        return;


    // If we got here, this means that we still have at least one ray active
    // This is a concurrent write by the way but we don't really care, everyone is writing
    // the same value
    render_data.aux_buffers.still_one_ray_active[0] = 1;

    render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer[pixel_index] = restir_gi_initial_candidate;
}

#endif
