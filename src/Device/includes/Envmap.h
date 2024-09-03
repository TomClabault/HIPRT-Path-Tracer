/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ENVMAP_H
#define DEVICE_ENVMAP_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
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

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F eval_environment_map_no_pdf(const WorldSettings& world_settings, const float3& direction)
{
    // Taking envmap rotation into account
    float3 rotated_direction = matrix_X_vec(world_settings.envmap_rotation_matrix, direction);;
    float u, v;

    u = 0.5f + atan2(rotated_direction.z, rotated_direction.x) / (2.0f * M_PI);
    v = 0.5f + asin(rotated_direction.y) / M_PI;

    return sample_environment_map_texture(world_settings, make_float2(u, 1.0f - v));
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
        int x_index = (lower + upper) / 2;
        int env_map_index = y_index * world_settings.envmap_width + x_index;

        if (value < world_settings.envmap_cdf[env_map_index])
            upper = x_index;
        else
            lower = x_index + 1;
    }
    x = hippt::max(hippt::min(lower, world_settings.envmap_width), 0u);
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F envmap_sample(const HIPRTRenderData& render_data, float3& sampled_direction, float& envmap_pdf, Xorshift32Generator& random_number_generator)
{
    const WorldSettings& world_settings = render_data.world_settings;

    // Importance sampling a texel of the envmap
    int x, y;
    unsigned int cdf_size = world_settings.envmap_width * world_settings.envmap_height;
    float env_map_total_sum = world_settings.envmap_cdf[cdf_size - 1];
    envmap_cdf_search(world_settings, random_number_generator() * env_map_total_sum, x, y);

    // Converting to UV coordinates
    float u = (float)x / world_settings.envmap_width;
    float v = (float)y / world_settings.envmap_height;

    // Converting to polar coordinates
    float phi = u * 2.0f * M_PI;
    // Clamping because a theta of 0.0f would mean straight up which means singularity
    // which means not good for numerical stability
    float theta = hippt::max(1.0e-5f, v * M_PI);

    // Convert to cartesian coordinates
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    sampled_direction = make_float3(-sin_theta * cos(phi), -cos_theta, -sin_theta * sin(phi));

    // Taking envmap rotation into account
    sampled_direction = matrix_X_vec(world_settings.envmap_rotation_matrix, sampled_direction);

    ColorRGB32F env_map_radiance = sample_environment_map_texture(world_settings, make_float2(u, 1.0f - v));
    envmap_pdf = luminance(env_map_radiance) / (env_map_total_sum * world_settings.envmap_intensity);
    envmap_pdf = (envmap_pdf * world_settings.envmap_width * world_settings.envmap_height) / (2.0f * M_PI * M_PI * sin_theta);

    return env_map_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F envmap_eval(const HIPRTRenderData& render_data, const float3& direction, float& pdf)
{
    const WorldSettings& world_settings = render_data.world_settings;

    ColorRGB32F envmap_radiance = eval_environment_map_no_pdf(world_settings, direction);

    unsigned int cdf_size = world_settings.envmap_width * world_settings.envmap_height;
    float envmap_total_sum = world_settings.envmap_cdf[cdf_size - 1];

    float theta_brdf_dir = acos(direction.z);
    float sin_theta = sin(theta_brdf_dir);

    // Probability of sampling that texel on the envmap
    pdf = envmap_radiance.luminance() / (envmap_total_sum * render_data.world_settings.envmap_intensity);
    pdf *= world_settings.envmap_width * world_settings.envmap_height;

    // Converting from "texel on envmap measure" to solid angle
    pdf /= (2.0f * M_PI * M_PI * sin_theta);

    return envmap_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_environment_map_cdf(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const RayVolumeState& volume_state, HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    const WorldSettings& world_settings = render_data.world_settings;

    float envmap_pdf;
    float3 sampled_direction;
    ColorRGB32F envmap_color = envmap_sample(render_data, sampled_direction, envmap_pdf, random_number_generator);
    ColorRGB32F envmap_mis_contribution;

    // Sampling the envmap with MIS
    float cosine_term = hippt::dot(closest_hit_info.shading_normal, sampled_direction);
    if (envmap_pdf > 0.0f && cosine_term > 0.0f)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
        shadow_ray.direction = sampled_direction;

        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e38f);
        if (!in_shadow)
        {
            float bsdf_pdf;
            float mis_weight;

            RayVolumeState trash_state = volume_state;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_state, view_direction, closest_hit_info.shading_normal, sampled_direction, bsdf_pdf);

            mis_weight = power_heuristic(envmap_pdf, bsdf_pdf);
            envmap_mis_contribution = bsdf_color * cosine_term * mis_weight * envmap_color / envmap_pdf;
        }
    }




    float brdf_sample_pdf;
    float3 bsdf_sampled_dir;
    RayVolumeState trash_state = volume_state;
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, material, trash_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bsdf_sampled_dir, brdf_sample_pdf, random_number_generator);
    ColorRGB32F bsdf_mis_contribution;

    // Sampling the BSDF with MIS
    cosine_term = hippt::clamp(0.0f, 1.0f, hippt::dot(closest_hit_info.shading_normal, bsdf_sampled_dir));
    if (brdf_sample_pdf > 0.0f && cosine_term > 0.0f)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-5f;
        shadow_ray.direction = bsdf_sampled_dir;

        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e38f);
        if (!in_shadow)
        {
            float envmap_pdf;
            ColorRGB32F envmap_color = envmap_eval(render_data, bsdf_sampled_dir, envmap_pdf);
            if (envmap_pdf > 0.0f)
            {
                float mis_weight;

                mis_weight = power_heuristic(brdf_sample_pdf, envmap_pdf);
                bsdf_mis_contribution = envmap_color * mis_weight * cosine_term * bsdf_color / brdf_sample_pdf;
            }
        }
    }

    return bsdf_mis_contribution + envmap_mis_contribution;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_environment_map(const HIPRTRenderData& render_data, const RayPayload& ray_payload, HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    const WorldSettings& world_settings = render_data.world_settings;

    if (world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // Not using the envmap
        return ColorRGB32F(0.0f);

    if (ray_payload.material.is_emissive())
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return ColorRGB32F(0.0f);

    if (world_settings.envmap_intensity <= 0.0f)
        // No need to sample the envmap if the user has set the intensity to 0
        return ColorRGB32F(0.0f);

#if EnvmapSamplingStrategy == ESS_BINARY_SEARCH
    return sample_environment_map_cdf(render_data, ray_payload.material, ray_payload.volume_state, closest_hit_info, view_direction, random_number_generator);
#elif EnvmapSamplingStrategy == ESS_NO_SAMPLING
    return ColorRGB32F(0.0f);
#endif
}

#endif
