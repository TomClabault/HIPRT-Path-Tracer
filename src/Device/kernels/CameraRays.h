/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
    if (render_data.aux_buffers.restir_di_reservoir_buffer_1 != nullptr)
    {
        // Only resetting if we're accumulating for offline rendering. Otherwise, we want the
        // temporal side of ReSTIR and so we're not resetting to at least keep the temporal
        // buffer alive
        //
        // Also, we're only resetting if the buffers aren't nullptr (they are nullptr 
        // if ReSTIR DI is currently disabled (using another direct lighting strategy).
        // We only need to check 1 buffer for that.

        if (render_data.aux_buffers.restir_di_reservoir_buffer_1)
            render_data.aux_buffers.restir_di_reservoir_buffer_1[pixel_index] = ReSTIRDIReservoir();

        if (render_data.aux_buffers.restir_di_reservoir_buffer_2)
            render_data.aux_buffers.restir_di_reservoir_buffer_2[pixel_index] = ReSTIRDIReservoir();

        if (render_data.aux_buffers.restir_di_reservoir_buffer_3)
            render_data.aux_buffers.restir_di_reservoir_buffer_3[pixel_index] = ReSTIRDIReservoir();
    }

    if (render_data.aux_buffers.restir_gi_reservoir_buffer_1 != nullptr)
    {
        // Same for ReSTIR GI
        if (render_data.aux_buffers.restir_gi_reservoir_buffer_1)
            render_data.aux_buffers.restir_gi_reservoir_buffer_1[pixel_index] = ReSTIRGIReservoir();

        if (render_data.aux_buffers.restir_gi_reservoir_buffer_2)
            render_data.aux_buffers.restir_gi_reservoir_buffer_2[pixel_index] = ReSTIRGIReservoir();

        if (render_data.aux_buffers.restir_gi_reservoir_buffer_3)
            render_data.aux_buffers.restir_gi_reservoir_buffer_3[pixel_index] = ReSTIRGIReservoir();
    }

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        // These buffers are only available when either the adaptive sampling or the stop noise threshold is enabled
        render_data.aux_buffers.pixel_sample_count[pixel_index] = 0;
        render_data.aux_buffers.pixel_squared_luminance[pixel_index] = 0;
        render_data.aux_buffers.pixel_converged_sample_count[pixel_index] = -1;
    }

    // Resetting the G-Buffer
    render_data.g_buffer.first_hit_prim_index[pixel_index] = -1;
    render_data.g_buffer.geometric_normals[pixel_index] = Octahedral24BitNormal::pack_static(make_float3(0.0f, 0.0f, 0.0f));
    render_data.g_buffer.shading_normals[pixel_index] = Octahedral24BitNormal::pack_static(make_float3(0.0f, 0.0f, 0.0f));
    render_data.g_buffer.primary_hit_position[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
    render_data.g_buffer.materials[pixel_index] = DevicePackedEffectiveMaterial::pack(DeviceUnpackedEffectiveMaterial());

    // Resetting the previous frame G-Buffer if we have it
    if (render_data.render_settings.use_prev_frame_g_buffer())
    {
        render_data.g_buffer_prev_frame.first_hit_prim_index[pixel_index] = -1;
        render_data.g_buffer_prev_frame.geometric_normals[pixel_index] = Octahedral24BitNormal::pack_static(make_float3(0.0f, 0.0f, 0.0f));
        render_data.g_buffer_prev_frame.shading_normals[pixel_index] = Octahedral24BitNormal::pack_static(make_float3(0.0f, 0.0f, 0.0f));
        render_data.g_buffer_prev_frame.primary_hit_position[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
        render_data.g_buffer_prev_frame.materials[pixel_index] = DevicePackedEffectiveMaterial::pack(DeviceUnpackedEffectiveMaterial());
    }
}

HIPRT_HOST_DEVICE HIPRT_INLINE void rescale_samples(HIPRTRenderData& render_data, uint32_t pixel_index)
{
    // Because when displaying the framebuffer, we're dividing by the number of samples to 
    // rescale the color of a pixel, we're going to have a problem if some pixels stopped samping
    // at 10 samples while the other pixels are still being sampled and have 100 samples for example. 
    // The pixels that only received 10 samples are going to be divided by 100 at display time, making them
    // appear too dark.
    // We're rescaling the color of the pixels that stopped sampling here for correct display

    float float_sample_number = static_cast<float>(render_data.render_settings.sample_number);
    render_data.buffers.accumulated_ray_colors[pixel_index] = render_data.buffers.accumulated_ray_colors[pixel_index] / float_sample_number * (render_data.render_settings.sample_number + 1);
    if (render_data.buffers.gmon_estimator.sets != nullptr)
    {
        int2 res = render_data.render_settings.render_resolution;
        // GMoN is enabled, we're also going to scale the GMoN samples for the same reason
        for (int set_index = 0; set_index < GMoNMSetsCount; set_index++)
            // TODO this is slow
            render_data.buffers.gmon_estimator.sets[set_index * res.x * res.y + pixel_index] = render_data.buffers.gmon_estimator.sets[set_index * res.x * res.y + pixel_index] / float_sample_number * (render_data.render_settings.sample_number + 1);
    }
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) CameraRays(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline CameraRays(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    if (render_data.render_settings.need_to_reset)
        reset_render(render_data, pixel_index);

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
        render_data.g_buffer_prev_frame.primary_hit_position[pixel_index] = render_data.g_buffer.primary_hit_position[pixel_index];
        render_data.g_buffer_prev_frame.first_hit_prim_index[pixel_index] = render_data.g_buffer.first_hit_prim_index[pixel_index];
    }

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
            hippt::atomic_fetch_add(render_data.aux_buffers.stop_noise_threshold_converged_count, 1u);
    }

    if (render_data.render_settings.has_access_to_adaptive_sampling_buffers())
    {
        if (!sampling_needed)
        {
            rescale_samples(render_data, pixel_index);

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
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
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

    hiprtRay ray = render_data.current_camera.get_camera_ray(x_ray_point_direction, y_ray_point_direction, render_data.render_settings.render_resolution);
    RayPayload ray_payload;

    HitInfo closest_hit_info;
    bool intersection_found = trace_main_path_ray(render_data, ray, ray_payload, closest_hit_info, /* camera ray = no previous primitive hit */ -1, /* bounce. Always 0 for camera rays*/ 0, random_number_generator);

    if (intersection_found)
    {
        render_data.g_buffer.geometric_normals[pixel_index].pack(closest_hit_info.geometric_normal);
        render_data.g_buffer.shading_normals[pixel_index].pack(closest_hit_info.shading_normal);

        render_data.g_buffer.materials[pixel_index] = DevicePackedEffectiveMaterial::pack(ray_payload.material);
        render_data.g_buffer.primary_hit_position[pixel_index] = closest_hit_info.inter_point;
    }
    else
        // Special case when not hitting anything
        //
        // The view directions are reconstructed from the primary hit and the camera position
        // but if we didn't hit anything, there's no primary hit. 
        // 
        // But we're still going to need to be able to reconstruct the view direction 
        // so we're faking the primary hit with the point the ray was directed to instead.
        //
        // If you're wondering: "yeah but then the rest of the ray tracing passes are going to use a wrong primary hit position?"
        //      --> No because the 'first_hit_prim_index' indicates whether we have a primary hit or not.
        //          If we don't have a primary hit, we're never going to use the float3 in the 'primary_hit_position'
        //          buffer as an actual position, 
        render_data.g_buffer.primary_hit_position[pixel_index] = ray.origin + ray.direction;
        
    render_data.g_buffer.first_hit_prim_index[pixel_index] = intersection_found ? closest_hit_info.primitive_index : -1;
    render_data.aux_buffers.pixel_active[pixel_index] = true;

    if (render_data.render_settings.regir_settings.grid_fill.representative_points_pixel_index != nullptr && render_data.render_settings.regir_settings.use_representative_points)
        // If we have ReGIR enabled, we're going to store in each cell the current intersection point (actually the pixel index)
        render_data.render_settings.regir_settings.store_representative_point_index(closest_hit_info.inter_point, pixel_index);

    // If we got here, this means that we still have at least one ray active
    if (render_data.render_settings.do_update_status_buffers)
    {
        // Updating if we have the right to (when do_update_status_buffers is true).
        // do_update_status_buffers is only true on the last sample of a frame
        render_data.aux_buffers.still_one_ray_active[0] = 1;
    }
}

#endif
