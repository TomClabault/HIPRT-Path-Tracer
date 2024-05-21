/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ENVMAP_H
#define DEVICE_ENVMAP_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/Texture.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_environment_map_from_direction(const WorldSettings& world_settings, const float3& direction)
{
    float u, v;
    u = 0.5f + atan2(direction.z, direction.x) / (2.0f * (float)M_PI);
    v = 0.5f + asin(direction.y) / (float)M_PI;

    return sample_texture_rgb(&world_settings.envmap, 0, make_int2(world_settings.envmap_width, world_settings.envmap_height), /* is_srgb */ false, make_float2(u, v));
}

HIPRT_HOST_DEVICE HIPRT_INLINE void env_map_cdf_search(const WorldSettings& world_settings, float value, int& x, int& y)
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

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB sample_environment_map(const HIPRTRenderData& render_data, const RendererMaterial& material, HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    if (render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // Not using the envmap
        return ColorRGB(0.0f);

    if (material.is_emissive())
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return ColorRGB(0.0f);

    // TODO we shouldn't need envmap sampling in the surface since we're going to fail the
    // visibility test anyway but this leads to darker transmissive surfaces for now. Why?
    //if (hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0.0f)
        // We're not direct sampling if we're inside a surface
        // 
        // We're using the geometric normal here because using the shading normal could lead
        // to false positive because of the black fringes when using smooth normals / normal mapping
        // + microfacet BRDFs
      //  return ColorRGB(0.0f);

    const WorldSettings& world_settings = render_data.world_settings;

    int x, y;
    unsigned int cdf_size = world_settings.envmap_width * world_settings.envmap_height;
    float env_map_total_sum = world_settings.envmap_cdf[cdf_size - 1];
    env_map_cdf_search(world_settings, random_number_generator() * env_map_total_sum, x, y);

    float u = (float)x / world_settings.envmap_width;
    float v = (float)y / world_settings.envmap_height;
    float phi = u * 2.0f * M_PI;
    // Clamping to avoid theta = 0 which would imply a skysphere direction straight up
    // which leads to a pdf of infinity since it is a singularity
    float theta = hippt::max(1.0e-5f, v * (float)M_PI);

    ColorRGB env_sample;
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);

    // Convert to cartesian coordinates
    float3 sampled_direction = make_float3(-sin_theta * cos(phi), -cos_theta, -sin_theta * sin(phi));

    float cosine_term = hippt::dot(closest_hit_info.shading_normal, sampled_direction);
    if (cosine_term > 0.0f)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
        shadow_ray.direction = sampled_direction;

        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e38f);
        if (!in_shadow)
        {
            ColorRGB pixel = sample_texture_rgb(&world_settings.envmap, 0, make_int2(world_settings.envmap_width, world_settings.envmap_height), /* is_srgb */ false, make_float2(u, v));
            float env_map_pdf = luminance(pixel) / env_map_total_sum;
            env_map_pdf = (env_map_pdf * world_settings.envmap_width * world_settings.envmap_height) / (2.0f * M_PI * M_PI * sin_theta);

            ColorRGB env_map_radiance = sample_texture_rgb(&world_settings.envmap, 0, make_int2(world_settings.envmap_width, world_settings.envmap_height), /* is_srgb */ false, make_float2(u, v));
            float pdf;
            RayVolumeState trash_state;
            ColorRGB brdf = brdf_dispatcher_eval(render_data.buffers.materials_buffer, material, trash_state, view_direction, closest_hit_info.shading_normal, sampled_direction, pdf);

            float mis_weight = power_heuristic(env_map_pdf, pdf);
            env_sample = brdf * cosine_term * mis_weight * env_map_radiance / env_map_pdf;
        }
    }

    float brdf_sample_pdf;
    float3 brdf_sampled_dir;
    RayVolumeState trash_state;
    ColorRGB brdf_imp_sampling = brdf_dispatcher_sample(render_data.buffers.materials_buffer, material, trash_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, brdf_sampled_dir, brdf_sample_pdf, random_number_generator);

    cosine_term = hippt::clamp(0.0f, 1.0f, hippt::dot(closest_hit_info.shading_normal, brdf_sampled_dir));
    ColorRGB brdf_sample;
    if (brdf_sample_pdf != 0.0f && cosine_term > 0.0f)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-5f;
        shadow_ray.direction = brdf_sampled_dir;

        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, 1.0e38f);
        if (!in_shadow)
        {
            ColorRGB skysphere_color = sample_environment_map_from_direction(world_settings, brdf_sampled_dir);
            float theta_brdf_dir = acos(brdf_sampled_dir.z);
            float sin_theta_bdrf_dir = sin(theta_brdf_dir);
            float env_map_pdf = skysphere_color.luminance() / env_map_total_sum;

            env_map_pdf *= world_settings.envmap_width * world_settings.envmap_height;
            env_map_pdf /= (2.0f * M_PI * M_PI * sin_theta_bdrf_dir);

            float mis_weight = power_heuristic(brdf_sample_pdf, env_map_pdf);
            brdf_sample = skysphere_color * mis_weight * cosine_term * brdf_imp_sampling / brdf_sample_pdf;
        }
    }

    return brdf_sample + env_sample;
}

#endif
