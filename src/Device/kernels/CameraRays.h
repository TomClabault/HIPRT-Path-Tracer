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

HIPRT_HOST_DEVICE HIPRT_INLINE void reset_render(const HIPRTRenderData& render_data, uint32_t pixel_index)
{
    if (render_data.render_settings.accumulate && render_data.aux_buffers.restir_reservoir_buffer_1 != nullptr)
    {
        // Only resetting if we're accumulating for offline rendering. Otherwise, we want the
        // temporal side of ReSTIR and so we're not resetting to at least keep the temporal
        // buffer alive
        //
        // Also, we're only resetting if the buffers aren't nullptr (they are nullptr 
        // if ReSTIR DI is currently disabled (using another direct lighting strategy).
        // We only need to check 1 buffer for that.

        render_data.aux_buffers.restir_reservoir_buffer_1[pixel_index] = ReSTIRDIReservoir();
        render_data.aux_buffers.restir_reservoir_buffer_2[pixel_index] = ReSTIRDIReservoir();
        render_data.aux_buffers.restir_reservoir_buffer_3[pixel_index] = ReSTIRDIReservoir();
    }

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        // These buffers are only available when either the adaptive sampling or the stop noise threshold is enabled
        render_data.aux_buffers.pixel_sample_count[pixel_index] = 0;
        render_data.aux_buffers.pixel_squared_luminance[pixel_index] = 0;
        render_data.aux_buffers.pixel_converged_sample_count[pixel_index] = -1;
    }
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) CameraRays(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline CameraRays(HIPRTRenderData render_data, int2 res, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= res.x || y >= res.y)
        return;

    uint32_t pixel_index = x + y * res.x;

    // 'Render low resolution' means that the user is moving the camera for example
    // so we're going to reduce the quality of the render for increased framerates
    // while moving
    if (render_data.render_settings.do_render_low_resolution())
    {
        int res_scaling = render_data.render_settings.render_low_resolution_scaling;

        // If rendering at low resolution, only one pixel out of res_scaling^2 will be rendered
        if (x % res_scaling != 0 || y % res_scaling != 0)
        {
            render_data.aux_buffers.pixel_active[pixel_index] = false;

            return;
        }

        pixel_index /= res_scaling;
    }

    if (render_data.render_settings.use_prev_frame_g_buffer())
    {
        render_data.g_buffer_prev_frame.geometric_normals[pixel_index] = render_data.g_buffer.geometric_normals[pixel_index];
        render_data.g_buffer_prev_frame.shading_normals[pixel_index] = render_data.g_buffer.shading_normals[pixel_index];
        render_data.g_buffer_prev_frame.materials[pixel_index] = render_data.g_buffer.materials[pixel_index];
        render_data.g_buffer_prev_frame.first_hits[pixel_index] = render_data.g_buffer.first_hits[pixel_index];
        render_data.g_buffer_prev_frame.first_hit_prim_index[pixel_index] = render_data.g_buffer.first_hit_prim_index[pixel_index];
        render_data.g_buffer_prev_frame.ray_volume_states[pixel_index] = render_data.g_buffer.ray_volume_states[pixel_index];
        render_data.g_buffer_prev_frame.camera_ray_hit[pixel_index] = render_data.g_buffer.camera_ray_hit[pixel_index];
    }

    if (render_data.render_settings.sample_number == 0 || render_data.render_settings.need_to_reset)
        reset_render(render_data, pixel_index);

    bool sampling_needed = true;
    bool pixel_converged = false;
    sampling_needed = adaptive_sampling(render_data, pixel_index, pixel_converged);
    
    if (pixel_converged || !sampling_needed)
    {
        if (render_data.render_settings.do_update_status_buffers)
            // Updating if we have the right to (when do_update_status_buffers is true).
            // do_update_status_buffers is only true on the last sample of a frame
            // 
            // Indicating that this pixel has reached the threshold in render_settings.stop_noise_threshold
            hippt::atomic_add(render_data.aux_buffers.stop_noise_threshold_converged_count, 1u);
    }

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        if (!sampling_needed)
        {
            // Because when displaying the framebuffer, we're dividing by the number of samples to 
            // rescale the color of a pixel, we're going to have a problem if some pixels stopped samping
            // at 10 samples while the other pixels are still being sampled and have 100 samples for example. 
            // The pixels that only received 10 samples are going to be divided by 100 at display time, making them
            // appear too dark.
            // We're rescaling the color of the pixels that stopped sampling here for correct display

            render_data.buffers.pixels[pixel_index] = render_data.buffers.pixels[pixel_index] / render_data.render_settings.sample_number * (render_data.render_settings.sample_number + 1);
            render_data.aux_buffers.pixel_active[pixel_index] = false;

            return;
        }
        else
            render_data.aux_buffers.pixel_sample_count[pixel_index]++;
    }

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);
    Xorshift32Generator random_number_generator(seed);

    // Direction to the center of the pixel
    float x_ray_point_direction = (x + 0.5f);
    float y_ray_point_direction = (y + 0.5f);
    if (render_data.current_camera.do_jittering)
    {
        // Jitter randomly around the center
        x_ray_point_direction += random_number_generator() - 0.5f;
        y_ray_point_direction += random_number_generator() - 0.5f;
    }

    hiprtRay ray = render_data.current_camera.get_camera_ray(x_ray_point_direction, y_ray_point_direction, res);
    RayPayload ray_payload;

    HitInfo closest_hit_info;
    bool intersection_found = trace_ray(render_data, ray, ray_payload, closest_hit_info, /* camera ray = no previous primitive hit */ -1, random_number_generator);

    if (intersection_found)
    {
        if (ray_payload.material.is_emissive() && hippt::dot(-ray.direction, closest_hit_info.geometric_normal) < 0)
        {
            // TODO necessary? Not already handled by trace_ray() ?
            closest_hit_info.geometric_normal = -closest_hit_info.geometric_normal;
            closest_hit_info.shading_normal = -closest_hit_info.shading_normal;
        }

        render_data.g_buffer.geometric_normals[pixel_index] = closest_hit_info.geometric_normal;
        render_data.g_buffer.shading_normals[pixel_index] = closest_hit_info.shading_normal;

        render_data.g_buffer.materials[pixel_index] = DevicePackedEffectiveMaterial::pack(ray_payload.material);
        render_data.g_buffer.first_hits[pixel_index] = closest_hit_info.inter_point;
        render_data.g_buffer.ray_volume_states[pixel_index] = ray_payload.volume_state;
    }
        
    render_data.g_buffer.first_hit_prim_index[pixel_index] = intersection_found ? closest_hit_info.primitive_index : -1;

    render_data.g_buffer.camera_ray_hit[pixel_index] = intersection_found;
    render_data.aux_buffers.pixel_active[pixel_index] = true;

    // If we got here, this means that we still have at least one ray active
    if (render_data.render_settings.do_update_status_buffers)
    {
        // Updating if we have the right to (when do_update_status_buffers is true).
        // do_update_status_buffers is only true on the last sample of a frame
        render_data.aux_buffers.still_one_ray_active[0] = 1;
    }
}

#endif
