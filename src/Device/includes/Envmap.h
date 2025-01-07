/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ENVMAP_H
#define DEVICE_ENVMAP_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/MISBSDFRayReuse.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/Texture.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

/**
 * References:
 * 
 * [1] [GLSL Path Tracer implementation by knightcrawler25] https://github.com/knightcrawler25/GLSL-PathTracer
 * [2] [PBR Book 3rd Ed - Infinite Light Sampling] https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources
 */ 

/**
 * This function expects 'direction' to be in world space
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F eval_envmap_no_pdf(const WorldSettings& world_settings, const float3& direction)
{
    // Bringing the direction in envmap space for sampling the envmap
    float3 rotated_direction = matrix_X_vec(world_settings.world_to_envmap_matrix, direction);

    float u = 0.5f + atan2(rotated_direction.z, rotated_direction.x) * M_INV_2_PI;
    float v = 0.5f + asin(rotated_direction.y) * M_INV_PI;

    return sample_environment_map_texture(world_settings, make_float2(u, v));
}

HIPRT_HOST_DEVICE HIPRT_INLINE void envmap_cdf_search(const WorldSettings& world_settings, float value, int& x, int& y)
{
    //First searching a line to sample
    unsigned int lower = 0;
    int upper = world_settings.envmap_height - 1;

    int x_index = world_settings.envmap_width - 1;
    while (lower < upper)
    {
        int y_index = (lower + upper) / 2;
        int env_map_index = y_index * world_settings.envmap_width + x_index;

        if (value < world_settings.envmap_cdf[env_map_index])
            upper = y_index;
        else
            lower = y_index + 1;
    }
    y = hippt::max(hippt::min(lower, world_settings.envmap_height), 0u);

    //Then sampling the line itself
    lower = 0;
    upper = world_settings.envmap_width - 1;

    int y_index = y;
    while (lower < upper)
    {
        int x_idx = (lower + upper) / 2;
        int env_map_index = y_index * world_settings.envmap_width + x_idx;

        if (value < world_settings.envmap_cdf[env_map_index])
            upper = x_idx;
        else
            lower = x_idx + 1;
    }
    x = hippt::max(hippt::min(lower, world_settings.envmap_width), 0u);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F envmap_sample(const WorldSettings& world_settings, float3& sampled_direction, float& envmap_pdf, Xorshift32Generator& random_number_generator)
{
    int x, y;
    float env_map_total_sum = world_settings.envmap_total_sum;

#if EnvmapSamplingStrategy == ESS_BINARY_SEARCH
    // Importance sampling a texel of the envmap with a binary search on the CDF
    envmap_cdf_search(world_settings, random_number_generator() * env_map_total_sum, x, y);
#else
    int random_index = random_number_generator.random_index(world_settings.envmap_height * world_settings.envmap_width);
    float probability = world_settings.alias_table_probas[random_index];
    if (random_number_generator() > probability)
        // Picking the alias
        random_index = world_settings.alias_table_alias[random_index];

    y = static_cast<int>(random_index / world_settings.envmap_width);
    x = static_cast<int>(random_index - y * world_settings.envmap_width);
#endif

    // Converting to UV coordinates
    float u = static_cast<float>(x) / world_settings.envmap_width;
    float v = static_cast<float>(y) / world_settings.envmap_height;

    // Converting to polar coordinates
    float phi = u * M_TWO_PI;
    // Clamping because a theta of 0.0f would mean straight up which means singularity
    // which means not good for numerical stability
    float theta = hippt::max(1.0e-5f, v * M_PI);

    // Convert to cartesian coordinates
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    // Using this formula here instead of the usual (sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)
    // because we want our envmap to be Y-up
    sampled_direction = make_float3(-sin_theta * cos(phi), -cos_theta, -sin_theta * sin(phi));

    // Taking envmap rotation into account to bring the direction in world space
    sampled_direction = matrix_X_vec(world_settings.envmap_to_world_matrix, sampled_direction);

    ColorRGB32F env_map_radiance = sample_environment_map_texture(world_settings, make_float2(u, v));
    // Computing envmap PDF
    envmap_pdf = env_map_radiance.luminance() / (env_map_total_sum * world_settings.envmap_intensity);
    envmap_pdf *= world_settings.envmap_width * world_settings.envmap_height;
    // Converting the PDF from area measure on the envmap to solid angle measure
    envmap_pdf /= (M_TWO_PIPI * sin_theta);

    return env_map_radiance;
}

/**
 * This function expects the given direction to be in world space i.e.
 * the direction is already rotated by the envmap rotation matrix
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F envmap_eval(const HIPRTRenderData& render_data, const float3& direction, float& pdf)
{
    const WorldSettings& world_settings = render_data.world_settings;

    ColorRGB32F envmap_radiance = eval_envmap_no_pdf(world_settings, direction);

    float envmap_total_sum = world_settings.envmap_total_sum;

    float theta_bsdf_dir = acos(-direction.y);
    float sin_theta = sin(theta_bsdf_dir);

    // Probability of sampling that texel on the envmap
    pdf = envmap_radiance.luminance() / (envmap_total_sum * render_data.world_settings.envmap_intensity);
    pdf *= world_settings.envmap_width * world_settings.envmap_height;

    // Converting from "texel on envmap measure" to solid angle
    pdf /= (M_TWO_PIPI * sin_theta);

    return envmap_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_environment_map_with_mis(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, RayVolumeState& volume_state, HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, MISBSDFRayReuse& mis_ray_reuse)
{
    float envmap_pdf;
    float3 sampled_direction;
    ColorRGB32F envmap_color = envmap_sample(render_data.world_settings, sampled_direction, envmap_pdf, random_number_generator);
    ColorRGB32F envmap_mis_contribution;

    // Sampling the envmap with MIS
    float cosine_term = hippt::dot(closest_hit_info.shading_normal, sampled_direction);
    if (envmap_pdf > 0.0f && cosine_term > 0.0f)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = closest_hit_info.inter_point;
        shadow_ray.direction = sampled_direction;

        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e35f, closest_hit_info.primitive_index, random_number_generator);
        if (!in_shadow)
        {
            float bsdf_pdf;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, material, volume_state, false, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction, bsdf_pdf, random_number_generator);

#if EnvmapSamplingDoBSDFMIS
            float mis_weight = balance_heuristic(envmap_pdf, bsdf_pdf);
#else
            float mis_weight = 1.0f;
#endif

            envmap_mis_contribution = bsdf_color * cosine_term * mis_weight * envmap_color / envmap_pdf;
        }
    }




#if EnvmapSamplingDoBSDFMIS
    float bsdf_sample_pdf;
    float3 bsdf_sampled_dir;
    ColorRGB32F bsdf_color;
    ColorRGB32F bsdf_mis_contribution;

#if ReuseBSDFMISRay == KERNEL_OPTION_TRUE
    if (mis_ray_reuse.has_ray())
    {
        // If we already have a BSDF ray to reuse from next-event estimation on the emissive lights,
        // let's reuse it.

        if (mis_ray_reuse.next_ray_state == RayState::MISSED)
        {
            // We only want to reuse rays that missed all geometry otherwise we can't see the envmap...
            bsdf_sample_pdf = mis_ray_reuse.bsdf_pdf;
            bsdf_sampled_dir = mis_ray_reuse.bsdf_sampled_direction;
            bsdf_color = mis_ray_reuse.bsdf_color;
        }
        else
            // If the ray that we're reusing doesn't see the envmap, let's just return
            // the envmap sample contribution (because the BSDF sample contribution is going
            // to be 0 anyways since this BSDF sample is occluded and can't see the envmap)
            return envmap_mis_contribution;
    }
    else
        // No BSDF MIS ray to reuse, let's sample the BSDF
        bsdf_color = bsdf_dispatcher_sample(render_data, material, volume_state, false, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bsdf_sampled_dir, bsdf_sample_pdf, random_number_generator);
#else
        // No BSDF MIS ray to reuse, let's sample the BSDF
        bsdf_color = bsdf_dispatcher_sample(render_data, material, volume_state, false, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bsdf_sampled_dir, bsdf_sample_pdf, random_number_generator);
#endif

    // Sampling the BSDF with MIS
    cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, bsdf_sampled_dir));
    if (bsdf_sample_pdf > 0.0f)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = closest_hit_info.inter_point;
        shadow_ray.direction = bsdf_sampled_dir;

#if ReuseBSDFMISRay == KERNEL_OPTION_TRUE
        bool in_shadow;
        if (mis_ray_reuse.has_ray() && mis_ray_reuse.next_ray_state == RayState::MISSED)
            // If we've reused a ray, we already know that it is not occluded
            in_shadow = false;
        else
            // No ray was reused, we have to check for visibility
            in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e35f, closest_hit_info.primitive_index, random_number_generator);
#else
        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e35f, closest_hit_info.primitive_index, random_number_generator);
#endif

        if (!in_shadow)
        {
            float envmap_eval_pdf;
            ColorRGB32F envmap_radiance = envmap_eval(render_data, bsdf_sampled_dir, envmap_eval_pdf);
            if (envmap_eval_pdf > 0.0f)
            {
                float mis_weight = balance_heuristic(bsdf_sample_pdf, envmap_eval_pdf);
                bsdf_mis_contribution = envmap_radiance * mis_weight * cosine_term * bsdf_color / bsdf_sample_pdf;
            }
        }
    }

    return bsdf_mis_contribution + envmap_mis_contribution;
#else
    return envmap_mis_contribution;
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_environment_map(const HIPRTRenderData& render_data, RayPayload& ray_payload, HitInfo& closest_hit_info, const float3& view_direction, int bounce, Xorshift32Generator& random_number_generator, MISBSDFRayReuse& mis_ray_reuse)
{
    const WorldSettings& world_settings = render_data.world_settings;

    if (world_settings.ambient_light_type != AmbientLightType::ENVMAP || render_data.bsdfs_data.white_furnace_mode)
        // Not using the envmap
        return ColorRGB32F(0.0f);

    if (ray_payload.material.is_emissive())
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return ColorRGB32F(0.0f);

    if (world_settings.envmap_intensity <= 0.0f)
        // No need to sample the envmap if the user has set the intensity to 0
        return ColorRGB32F(0.0f);

    if (bounce == 0 && DirectLightSamplingStrategy == LSS_RESTIR_DI)
        // The envmap lighting is handled by ReSTIR DI on the first bounce
        return ColorRGB32F(0.0f);

#if EnvmapSamplingStrategy == ESS_NO_SAMPLING
    return ColorRGB32F(0.0f);
#else
    return sample_environment_map_with_mis(render_data, ray_payload.material, ray_payload.volume_state, closest_hit_info, view_direction, random_number_generator, mis_ray_reuse);
#endif
}

#endif
