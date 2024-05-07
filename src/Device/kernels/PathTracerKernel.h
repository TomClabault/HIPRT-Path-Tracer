/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/AdaptiveSampling.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Lights.h"
#include "Device/includes/Envmap.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/Camera.h"
#include "HostDeviceCommon/Xorshift.h"

#define LOW_RESOLUTION_RENDER_DOWNSCALE 8

HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void debug_set_final_color(const HIPRTRenderData& render_data, int x, int y, int res_x, ColorRGB final_color)
{
    if (render_data.render_settings.sample_number == 0)
        render_data.buffers.pixels[y * res_x + x] = final_color;
    else
        render_data.buffers.pixels[y * res_x + x] = render_data.buffers.pixels[y * res_x + x] + final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_negative_color(ColorRGB sample_color, int x, int y, int sample)
{
    if (sample_color.r < 0 || sample_color.g < 0 || sample_color.b < 0)
    {
#ifndef __KERNELCC__
        std::cout << "Negative color at [" << x << ", " << y << "], sample " << sample << std::endl;
#endif

        return true;
    }

    return false;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_for_nan(ColorRGB sample_color, int x, int y, int sample)
{
    if (hippt::isNaN(sample_color.r) || hippt::isNaN(sample_color.g) || hippt::isNaN(sample_color.b))
    {
#ifndef __KERNELCC__
        std::cout << "NaN at [" << x << ", " << y << "], sample" << sample << std::endl;
#endif
        return true;
    }

    return false;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) PathTracerKernel(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline PathTracerKernel(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    const uint32_t index = (x + y * res.x);

    if (index >= res.x * res.y)
        return;

    // 'Render low resolution' means that the user is moving the camera for example
    // so we're going to reduce the quality of the render for increased framerates
    // while moving
    if (render_data.render_settings.render_low_resolution)
    {
        // Reducing the number of bounces to 3
        render_data.render_settings.nb_bounces = 3;
        render_data.render_settings.samples_per_frame = 1;

        // If rendering at low resolution, only one pixel out of 
        // LOW_RESOLUTION_RENDER_DOWNSCALE x LOW_RESOLUTION_RENDER_DOWNSCALE will be rendered
        if (x & (LOW_RESOLUTION_RENDER_DOWNSCALE - 1) || y & (LOW_RESOLUTION_RENDER_DOWNSCALE - 1))
            return;
    }

    if (render_data.render_settings.sample_number == 0)
    {
        // Resetting all buffers on the first frame
        render_data.buffers.pixels[index] = ColorRGB(0.0f);
        render_data.aux_buffers.denoiser_normals[index] = make_float3(1.0f, 1.0f, 1.0f);
        render_data.aux_buffers.denoiser_albedo[index] = ColorRGB(0.0f, 0.0f, 0.0f);
        render_data.aux_buffers.pixel_sample_count[index] = 0;
        render_data.aux_buffers.pixel_squared_luminance[index] = 0;
    }

    bool sampling_needed = true;
    if (render_data.render_settings.enable_adaptive_sampling)
        sampling_needed = adaptive_sampling(render_data, index);

    if (!sampling_needed)
    {
        // Because when displaying the framebuffer, we're dividing by the number of samples to 
        // rescale the color of a pixel, we're going to have a problem if some pixels stopped samping
        // at 10 samples while the other pixels are still being sampled and have 100 samples for example. 
        // The pixels that only received 10 samples are going to be divided by 100 at display time, making them
        // appear too dark.
        // We're rescaling the color of the pixels that stopped sampling here for correct display
        render_data.buffers.pixels[index] = render_data.buffers.pixels[index] / render_data.render_settings.sample_number * (render_data.render_settings.sample_number + render_data.render_settings.samples_per_frame);
        render_data.aux_buffers.debug_pixel_active[index] = 0;
        return;
    }
    else
        render_data.aux_buffers.debug_pixel_active[index] = render_data.render_settings.sample_number;

    Xorshift32Generator random_number_generator(wang_hash((index + 1) * (render_data.render_settings.sample_number + 1)));

    float squared_luminance_of_samples = 0.0f;
    ColorRGB final_color = ColorRGB(0.0f, 0.0f, 0.0f);
    ColorRGB denoiser_albedo = ColorRGB(0.0f, 0.0f, 0.0f);
    float3 denoiser_normal = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < render_data.render_settings.samples_per_frame; sample++)
    {
        //Jittered around the center
        float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
        float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;

        hiprtRay ray = camera.get_camera_ray(x_jittered, y_jittered, res);

        ColorRGB throughput = ColorRGB(1.0f);
        ColorRGB sample_color = ColorRGB(0.0f);
        RayState next_ray_state = RayState::BOUNCE;
        BRDF last_brdf_hit_type = BRDF::Uninitialized;

        // Whether or not we've already written to the denoiser's buffers
        bool denoiser_AOVs_set = false;
        float denoiser_blend = 1.0f;

        for (int bounce = 0; bounce < render_data.render_settings.nb_bounces; bounce++)
        {
            if (next_ray_state == RayState::BOUNCE)
            {
                HitInfo closest_hit_info;
                bool intersection_found = trace_ray(render_data, ray, closest_hit_info);

                if (intersection_found)
                {
                    int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
                    RendererMaterial material = render_data.buffers.materials_buffer[material_index];
                    last_brdf_hit_type = material.brdf_type;

                    // For the BRDF calculations, bounces, ... to be correct, we need the normal to be in the same hemisphere as
                    // the view direction. One thing that can go wrong is when we have an emissive triangle (typical area light)
                    // and a ray hits the back of the triangle. The normal will not be facing the view direction in this
                    // case and this will cause issues later in the BRDF.
                    // Because we want to allow backfacing emissive geometry (making the emissive geometry double sided
                    // and emitting light in both directions of the surface), we're negating the normal to make
                    // it face the view direction (but only for emissive geometry)
                    if (material.is_emissive() && hippt::dot(-ray.direction, closest_hit_info.geometric_normal) < 0)
                    {
                        closest_hit_info.geometric_normal = -closest_hit_info.geometric_normal;
                        closest_hit_info.shading_normal = -closest_hit_info.shading_normal;
                    }

                    // --------------------------------------------------- //
                    // ----------------- Direct lighting ----------------- //
                    // --------------------------------------------------- //
                    ColorRGB light_sample_radiance = sample_light_sources(render_data, -ray.direction, closest_hit_info, material, random_number_generator);
                    ColorRGB env_map_radiance = render_data.world_settings.use_ambient_light ? ColorRGB(0.0f) : sample_environment_map(render_data, -ray.direction, closest_hit_info, material, random_number_generator);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    // 0.003
                    float brdf_pdf;
                    float3 bounce_direction;
                    ColorRGB brdf = brdf_dispatcher_sample(material, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bounce_direction, brdf_pdf, random_number_generator);

                    if (last_brdf_hit_type == BRDF::SpecularFresnel)
                        // The fresnel blend coefficient is in the PDF
                        denoiser_blend *= brdf_pdf;

                    if (!denoiser_AOVs_set && last_brdf_hit_type != BRDF::SpecularFresnel)
                    {
                        denoiser_AOVs_set = true;

                        denoiser_albedo += material.base_color * denoiser_blend;
                        denoiser_normal += closest_hit_info.shading_normal * denoiser_blend;
                    }

                    // Terminate ray if something went wrong according to the unforgivable laws of physic
                    // (sampling a direction below the surface for example)
                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf <= 0.0f)
                        break;

                    if (bounce == 0)
                        sample_color = sample_color + material.emission * throughput;
                    sample_color = sample_color + (light_sample_radiance + env_map_radiance) * throughput;

                    throughput *= brdf * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal)) / brdf_pdf;

                    int outside_surface = hippt::dot(bounce_direction, closest_hit_info.shading_normal) < 0 ? -1.0f : 1.0;
                    float3 new_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 3.0e-3f * outside_surface;
                    ray.origin = new_ray_origin;
                    ray.direction = bounce_direction;

                    next_ray_state = RayState::BOUNCE;
                }
                else
                {
                    ColorRGB skysphere_color;
                    if (render_data.world_settings.use_ambient_light)
                        skysphere_color = render_data.world_settings.ambient_light_color;
                    else if (bounce == 0 || last_brdf_hit_type == BRDF::SpecularFresnel)
                    {
                        // We're only getting the skysphere radiance for the first rays because the
                        // syksphere is importance sampled.
                        // 
                        // We're also getting the skysphere radiance for perfectly specular BRDF since those
                        // are not importance sampled.

                        skysphere_color = sample_environment_map_from_direction(render_data.world_settings, ray.direction);
                    }

                    sample_color += skysphere_color * throughput;
                    next_ray_state = RayState::MISSED;
                }
            }
            else if (next_ray_state == RayState::MISSED)
                break;
        }

        // These 2 if() are basically anomally detectors.
        // They will set pixels to very bright colors if somehow
        // weird samples are produced
        // This helps spot unrobustness in the renderer 
        //
        // - Pink : sample with negative color
        // - Yellow : NaN sample
        bool invalid = false;
        invalid |= check_for_negative_color(sample_color, x, y, sample);
        invalid |= check_for_nan(sample_color, x, y, sample);

        if (invalid)
        {
            debug_set_final_color(render_data, x, y, res.x, ColorRGB(10000.0f, 0.0f, 10000.0f));

            return;
        }

        squared_luminance_of_samples += sample_color.luminance() * sample_color.luminance();
        final_color += sample_color;
    }

    render_data.buffers.pixels[index] += final_color;
    render_data.aux_buffers.pixel_squared_luminance[index] += squared_luminance_of_samples;
    render_data.aux_buffers.pixel_sample_count[index] += render_data.render_settings.samples_per_frame;

    // Handling denoiser's albedo and normals AOVs    
    denoiser_albedo /= (float)render_data.render_settings.samples_per_frame;
    denoiser_normal /= (float)render_data.render_settings.samples_per_frame;

    render_data.aux_buffers.denoiser_albedo[index] = (render_data.aux_buffers.denoiser_albedo[index] * render_data.render_settings.frame_number + denoiser_albedo) / (render_data.render_settings.frame_number + 1.0f);

    float3 accumulated_normal = (render_data.aux_buffers.denoiser_normals[index] * render_data.render_settings.frame_number + denoiser_normal) / (render_data.render_settings.frame_number + 1.0f);
    float normal_length = hippt::length(accumulated_normal);
    if (normal_length != 0.0f)
        // Checking that it is non-zero otherwise we would accumulate a persistent NaN in the buffer when normalizing by the 0-length
        render_data.aux_buffers.denoiser_normals[index] = accumulated_normal / normal_length;

    // TODO have that in the display shader
    // Handling low resolution render
    // The framebuffer actually still is at full resolution, it's just that we cast
    // 1 ray every 4, 8 or 16 pixels (depending on the low resolution factor)
    // This means that we have "holes" in the rendered where rays will never be cast
    // this loop fills the wholes by copying the pixel that we rendered to its unrendered
    // neighbors
    if (render_data.render_settings.render_low_resolution)
    {
        // Copying the pixel we just rendered to the neighbors
        for (int _y = 0; _y < LOW_RESOLUTION_RENDER_DOWNSCALE; _y++)
        {
            for (int _x = 0; _x < LOW_RESOLUTION_RENDER_DOWNSCALE; _x++)
            {
                int _index = _y * res.x + _x + index;
                if (_y == 0 && _x == 0)
                    // This is ourselves
                    continue;
                else if (_index >= res.x * res.y)
                    // Outside of the framebuffer
                    return;
                else
                {
                    // Actually a valid pixel
                    render_data.buffers.pixels[_index] = render_data.buffers.pixels[index];

                    // Also handling the denoiser AOVs. Useful only when the user is moving the camera
                    // (and thus rendering at low resolution) while the denoiser's normals / albedo has
                    // been selected as the active viewport view
                    render_data.aux_buffers.denoiser_albedo[_index] = render_data.aux_buffers.denoiser_albedo[index];
                    render_data.aux_buffers.denoiser_normals[_index] = render_data.aux_buffers.denoiser_normals[index];
                }
            }
        }
    }
}