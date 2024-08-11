/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_FULL_PATH_TRACER_H
#define KERNELS_FULL_PATH_TRACER_H

#include "Device/includes/AdaptiveSampling.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Lights.h"
#include "Device/includes/Envmap.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Material.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_HOST_DEVICE HIPRT_INLINE void debug_set_final_color(const HIPRTRenderData& render_data, int x, int y, int res_x, ColorRGB32F final_color)
{
    if (render_data.render_settings.sample_number == 0)
        render_data.buffers.pixels[y * res_x + x] = final_color;
    else
        render_data.buffers.pixels[y * res_x + x] = final_color * render_data.render_settings.sample_number;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_negative_color(ColorRGB32F ray_color, int x, int y, int sample)
{
    if (ray_color.r < 0 || ray_color.g < 0 || ray_color.b < 0)
    {
#ifndef __KERNELCC__
        std::cout << "Negative color at [" << x << ", " << y << "], sample " << sample << std::endl;
#endif

        return true;
    }

    return false;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_nan(ColorRGB32F ray_color, int x, int y, int sample)
{
    if (hippt::isNaN(ray_color.r) || hippt::isNaN(ray_color.g) || hippt::isNaN(ray_color.b))
    {
#ifndef __KERNELCC__
        std::cout << "NaN at [" << x << ", " << y << "], sample" << sample << std::endl;
#endif
        return true;
    }

    return false;
}

#ifndef __KERNELCC__
#include "Utils/Utils.h" // For debugbreak in sanity_check()
#endif
HIPRT_HOST_DEVICE HIPRT_INLINE bool sanity_check(const HIPRTRenderData& render_data, RayPayload& ray_payload, int x, int y, int2& res, int sample)
{
    bool invalid = false;
    invalid |= check_for_negative_color(ray_payload.ray_color, x, y, sample);
    invalid |= check_for_nan(ray_payload.ray_color, x, y, sample);

    if (invalid)
    {
        if (render_data.render_settings.display_NaNs)
            debug_set_final_color(render_data, x, y, res.x, ColorRGB32F(1.0e15f, 0.0f, 1.0e15f));
        else
            ray_payload.ray_color = ColorRGB32F(0.0f);
#ifndef __KERNELCC__
        Utils::debugbreak();
#endif
    }

    return !invalid;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void reset_render(const HIPRTRenderData& render_data, uint32_t pixel_index)
{
    // Resetting all buffers on the first frame
    render_data.buffers.pixels[pixel_index] = ColorRGB32F(0.0f);
    render_data.aux_buffers.denoiser_normals[pixel_index] = make_float3(1.0f, 1.0f, 1.0f);
    render_data.aux_buffers.denoiser_albedo[pixel_index] = ColorRGB32F(0.0f, 0.0f, 0.0f);

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        // These buffers are only available when either the adaptive sampling or the stop noise threshold is enabled
        render_data.aux_buffers.pixel_sample_count[pixel_index] = 0;
        render_data.aux_buffers.pixel_squared_luminance[pixel_index] = 0;
    }
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) FullPathTracer(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline FullPathTracer(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    uint32_t pixel_index = (x + y * res.x);
    if (pixel_index >= res.x * res.y)
        return;
    else if (!render_data.aux_buffers.pixel_active[pixel_index])
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1));
    Xorshift32Generator random_number_generator(seed);

    float squared_luminance_of_samples = 0.0f;
    ColorRGB32F final_color = ColorRGB32F(0.0f, 0.0f, 0.0f);
    ColorRGB32F denoiser_albedo = ColorRGB32F(0.0f, 0.0f, 0.0f);
    float3 denoiser_normal = make_float3(0.0f, 0.0f, 0.0f);

    // Initializing the closest hit info the information from the camera ray pass
    HitInfo closest_hit_info;
    closest_hit_info.inter_point = render_data.g_buffer.first_hits[pixel_index];
    closest_hit_info.geometric_normal = hippt::normalize(render_data.g_buffer.geometric_normals[pixel_index]);
    closest_hit_info.shading_normal = hippt::normalize(render_data.g_buffer.shading_normals[pixel_index]);

    // Initializing the ray with the information from the camera ray pass
    hiprtRay ray;
    ray.direction = hippt::normalize(-render_data.g_buffer.view_directions[pixel_index]);

    bool intersection_found = render_data.g_buffer.camera_ray_hit[pixel_index] == 1;

    RayPayload ray_payload;
    ray_payload.next_ray_state = RayState::BOUNCE;
    ray_payload.material = RendererMaterial(render_data.g_buffer.materials[pixel_index]);
    ray_payload.volume_state = render_data.g_buffer.ray_volume_states[pixel_index];

    // TODO fix samples per frame not working since we separated the camera ray pass
    // for (int sample = 0; sample < render_data.render_settings.samples_per_frame; sample++)
    for (int sample = 0; sample < 1; sample++)
    {
        for (int bounce = 0; bounce < render_data.render_settings.nb_bounces; bounce++)
        {
            if (ray_payload.next_ray_state == RayState::BOUNCE)
            {
                if (bounce > 0)
                {
                    // Not tracing for the primary ray because this has already been done in the camera ray pass

                    intersection_found = trace_ray(render_data, ray, ray_payload, closest_hit_info);
                }

                if (intersection_found)
                {
                    if (bounce == 0)
                    {
                        denoiser_normal += closest_hit_info.shading_normal;
                        denoiser_albedo += ray_payload.material.base_color;
                    }

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

                    ColorRGB32F light_direct_contribution = sample_one_light(render_data, ray_payload, closest_hit_info, -ray.direction, random_number_generator, make_int2(x, y), res, bounce);
                    ColorRGB32F envmap_direct_contribution = sample_environment_map(render_data, ray_payload, closest_hit_info, -ray.direction, random_number_generator);

                    // Clamping direct lighting
                    light_direct_contribution = clamp_light_contribution(light_direct_contribution, render_data.render_settings.direct_contribution_clamp, bounce == 0);
                    envmap_direct_contribution = clamp_light_contribution(envmap_direct_contribution, render_data.render_settings.envmap_contribution_clamp, bounce == 0);

                    // Clamping indirect lighting 
                    light_direct_contribution = clamp_light_contribution(light_direct_contribution, render_data.render_settings.indirect_contribution_clamp, bounce > 0);
                    envmap_direct_contribution = clamp_light_contribution(envmap_direct_contribution, render_data.render_settings.indirect_contribution_clamp, bounce > 0);

#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING // No direct light sampling
                    ColorRGB32F hit_emission = ray_payload.material.emission;
                    hit_emission = clamp_light_contribution(hit_emission, render_data.render_settings.indirect_contribution_clamp, bounce > 0);

                    ray_payload.ray_color += hit_emission * ray_payload.throughput;
#else
                    if (bounce == 0)
                        // If we do have emissive geometry sampling, we only want to take
                        // it into account on the first bounce, otherwise we would be
                        // accounting for direct light sampling twice (bounce on emissive
                        // geometry + direct light sampling). Otherwise, we don't check for bounce == 0
                        ray_payload.ray_color += ray_payload.material.emission * ray_payload.throughput;
#endif

                    ray_payload.ray_color += (light_direct_contribution + envmap_direct_contribution) * ray_payload.throughput;

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    float3 bounce_direction;
                    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, ray_payload.volume_state, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bounce_direction, brdf_pdf, random_number_generator);

                    // Terminate ray if bad sampling
                    if (brdf_pdf <= 0.0f)
                        break;

                    int outside_surface = hippt::dot(bounce_direction, closest_hit_info.shading_normal) < 0 ? -1.0f : 1.0f;
                    ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 3.0e-3f * outside_surface;
                    ray.direction = bounce_direction;

                    ray_payload.throughput *= bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal)) / brdf_pdf;
                    ray_payload.next_ray_state = RayState::BOUNCE;
                }
                else
                {
                    ColorRGB32F skysphere_color;
                    if (render_data.world_settings.ambient_light_type == AmbientLightType::UNIFORM)
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
                            // 
                            // We're also getting the skysphere radiance for perfectly specular BRDF since those
                            // are not importance sampled.

                            skysphere_color = sample_environment_map_from_direction(render_data.world_settings, ray.direction);

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

                    skysphere_color = clamp_light_contribution(skysphere_color, render_data.render_settings.envmap_contribution_clamp, true);

                    ray_payload.ray_color += skysphere_color * ray_payload.throughput;
                    ray_payload.next_ray_state = RayState::MISSED;
                }
            }
            else if (ray_payload.next_ray_state == RayState::MISSED)
                break;
        }

        // Checking for NaNs / negative value samples. Output 
        if (!sanity_check(render_data, ray_payload, x, y, res, sample))
            return;

        squared_luminance_of_samples += ray_payload.ray_color.luminance() * ray_payload.ray_color.luminance();
        final_color += ray_payload.ray_color;
    }

    // If we got here, this means that we still have at least one ray active
    render_data.aux_buffers.still_one_ray_active[0] = 1;

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        // We can only use these buffers if the adaptive sampling or the stop noise threshold is enabled.
        // Otherwise, the buffers are destroyed to save some VRAM so they are not accessible
        render_data.aux_buffers.pixel_squared_luminance[pixel_index] += squared_luminance_of_samples;

        // TODO fix if we can have more than 1 samples per frame since passes refactor
        render_data.aux_buffers.pixel_sample_count[pixel_index]++;
    }

    render_data.buffers.pixels[pixel_index] += final_color;

    // Handling denoiser's albedo and normals AOVs    
    denoiser_albedo /= (float)render_data.render_settings.samples_per_frame;
    denoiser_normal /= (float)render_data.render_settings.samples_per_frame;

    render_data.aux_buffers.denoiser_albedo[pixel_index] = (render_data.aux_buffers.denoiser_albedo[pixel_index] * render_data.render_settings.frame_number + denoiser_albedo) / (render_data.render_settings.frame_number + 1.0f);

    float3 accumulated_normal = (render_data.aux_buffers.denoiser_normals[pixel_index] * render_data.render_settings.frame_number + denoiser_normal) / (render_data.render_settings.frame_number + 1.0f);
    float normal_length = hippt::length(accumulated_normal);
    if (normal_length != 0.0f)
        // Checking that it is non-zero otherwise we would accumulate a persistent NaN in the buffer when normalizing by the 0-length
        render_data.aux_buffers.denoiser_normals[pixel_index] = accumulated_normal / normal_length;
}

#endif
