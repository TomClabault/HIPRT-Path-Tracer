/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_CAMERA_RAY_H
#define KERNELS_CAMERA_RAY_H

#include "Device/includes/AdaptiveSampling.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/RayPayload.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) CameraRays(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline CameraRays(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    uint32_t pixel_index = (x + y * res.x);
    if (pixel_index >= res.x * res.y)
        return;

    // 'Render low resolution' means that the user is moving the camera for example
    // so we're going to reduce the quality of the render for increased framerates
    // while moving
    if (render_data.render_settings.render_low_resolution)
    {
        // Reducing the number of bounces to 3
        render_data.render_settings.nb_bounces = 3;
        render_data.render_settings.samples_per_frame = 1;
        int res_scaling = render_data.render_settings.render_low_resolution_scaling;
        pixel_index /= res_scaling;

        // If rendering at low resolution, only one pixel out of res_scaling^2 will be rendered
        if (x % res_scaling != 0 || y % res_scaling != 0)
        {
            render_data.aux_buffers.pixel_active[pixel_index] = false;

            return;
        }
    }

    if (render_data.render_settings.sample_number == 0)
    {
        // Resetting all buffers on the first frame
        render_data.buffers.pixels[pixel_index] = ColorRGB32F(0.0f);
        render_data.aux_buffers.denoiser_normals[pixel_index] = make_float3(1.0f, 1.0f, 1.0f);
        render_data.aux_buffers.denoiser_albedo[pixel_index] = ColorRGB32F(0.0f, 0.0f, 0.0f);
        render_data.aux_buffers.initial_reservoirs[pixel_index] = Reservoir();
        render_data.aux_buffers.spatial_reservoirs[pixel_index] = Reservoir();

        render_data.g_buffer.geometric_normals[pixel_index] = { 0, 0, 0 };
        render_data.g_buffer.shading_normals[pixel_index] = { 0, 0, 0 };
        render_data.g_buffer.materials[pixel_index] = SimplifiedRendererMaterial();
        render_data.g_buffer.first_hits[pixel_index] = { 0, 0, 0 };
        render_data.g_buffer.ray_volume_states[pixel_index] = RayVolumeState();
        render_data.g_buffer.view_directions[pixel_index] = { 0, 0, 0 };
        render_data.g_buffer.camera_ray_hit[pixel_index] = false;
        render_data.aux_buffers.pixel_active[pixel_index] = false;

        if (render_data.render_settings.stop_pixel_noise_threshold > 0.0f || render_data.render_settings.enable_adaptive_sampling)
        {
            // These buffers are only available when either the adaptive sampling or the stop noise threshold is enabled
            render_data.aux_buffers.pixel_sample_count[pixel_index] = 0;
            render_data.aux_buffers.pixel_squared_luminance[pixel_index] = 0;
        }
    }


    bool sampling_needed = true;
    bool pixel_converged = false;
    sampling_needed = adaptive_sampling(render_data, pixel_index, pixel_converged);

    if (pixel_converged || !sampling_needed)
        // Indicating that this pixel has reached the threshold in render_settings.stop_noise_threshold
        hippt::atomic_add(render_data.aux_buffers.stop_noise_threshold_count, 1u);

    if (pixel_converged || !sampling_needed)
    {
        // Because when displaying the framebuffer, we're dividing by the number of samples to 
        // rescale the color of a pixel, we're going to have a problem if some pixels stopped samping
        // at 10 samples while the other pixels are still being sampled and have 100 samples for example. 
        // The pixels that only received 10 samples are going to be divided by 100 at display time, making them
        // appear too dark.
        // We're rescaling the color of the pixels that stopped sampling here for correct display

        // TODO + 1 at the end should + samples per frame if we can fix the number of samples per frame to be > 1 with the pass refactor
        render_data.buffers.pixels[pixel_index] = render_data.buffers.pixels[pixel_index] / render_data.render_settings.sample_number * (render_data.render_settings.sample_number + 1);
        render_data.aux_buffers.pixel_active[pixel_index] = false;

        return;
    }

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1));
    Xorshift32Generator random_number_generator(seed);

    //Jittered around the center
    float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
    float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;

    hiprtRay ray = camera.get_camera_ray(x_jittered, y_jittered, res);
    RayPayload ray_payload;

    HitInfo closest_hit_info;
    bool intersection_found = trace_ray(render_data, ray, ray_payload, closest_hit_info);

    if (intersection_found)
    {
        if (ray_payload.material.is_emissive() && hippt::dot(-ray.direction, closest_hit_info.geometric_normal) < 0)
        {
            closest_hit_info.geometric_normal = -closest_hit_info.geometric_normal;
            closest_hit_info.shading_normal = -closest_hit_info.shading_normal;
        }

        render_data.g_buffer.geometric_normals[pixel_index] = closest_hit_info.geometric_normal;
        render_data.g_buffer.shading_normals[pixel_index] = closest_hit_info.shading_normal;
        render_data.g_buffer.materials[pixel_index] = ray_payload.material.get_simplified_material();
        render_data.g_buffer.first_hits[pixel_index] = closest_hit_info.inter_point;
        render_data.g_buffer.ray_volume_states[pixel_index] = ray_payload.volume_state;
    }

    render_data.g_buffer.view_directions[pixel_index] = -ray.direction;
    render_data.g_buffer.camera_ray_hit[pixel_index] = intersection_found;
    render_data.aux_buffers.pixel_active[pixel_index] = true;

    // If we got here, this means that we still have at least one ray active
    render_data.aux_buffers.still_one_ray_active[0] = 1;
}

#endif
