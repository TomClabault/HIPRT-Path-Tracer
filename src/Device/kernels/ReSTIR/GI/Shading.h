/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_SHADING_H
#define KERNELS_RESTIR_GI_SHADING_H

#include "Device/includes/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Lights.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/TargetFunction.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_Shading(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_Shading(HIPRTRenderData render_data, int x, int y)
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

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator random_number_generator(seed);

    hiprtRay ray;
    ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    HitInfo closest_hit_info;
    closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];
    if (closest_hit_info.primitive_index == -1)
    {
        // Geometry miss, directly into the envmap
        ColorRGB32F envmap_radiance = path_tracing_miss_gather_envmap(render_data, ColorRGB32F(1.0f), ray.direction, 0, pixel_index);
        path_tracing_accumulate_color(render_data, envmap_radiance, pixel_index);

        return;
    }

    closest_hit_info.inter_point = render_data.g_buffer.primary_hit_position[pixel_index];
    closest_hit_info.shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();

    // Initializing the ray with the information from the camera ray pass
    RayPayload ray_payload;
    ray_payload.next_ray_state = RayState::BOUNCE;
    // Loading the first hit in the ray payload
    ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();
    ray_payload.volume_state.reconstruct_first_hit(
        ray_payload.material,
        render_data.buffers.material_indices,
        closest_hit_info.primitive_index,
        random_number_generator);

    float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    ColorRGB32F camera_outgoing_radiance;
    ColorRGB32F DEBUG_COLOR;
    ColorRGB32F DEBUG_FIRST_HIT_THROUGHPUT;
    ColorRGB32F DEBUG_FIRST_BSDF_COLOR_DOT;

    // Dummy mis_reuse variable
    MISBSDFRayReuse mis_reuse;
    if (render_data.render_settings.enable_direct)
        // Adding the direct lighting contribution at the first hit in the direction of the camera
        camera_outgoing_radiance += estimate_direct_lighting(render_data, ray_payload, closest_hit_info, view_direction, x, y, mis_reuse, random_number_generator);

    ReSTIRGIReservoir resampling_reservoir = render_data.render_settings.restir_gi_settings.restir_output_reservoirs[pixel_index];

    if (render_data.render_settings.nb_bounces > 0)
    {
        // Only doing the ReSTIR GI stuff if we have more than 1 bounce

        if (resampling_reservoir.UCW > 0.0f)
        {
            // Only doing the shading if we do actually have a sample

            float3 geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();

            float3 restir_resampled_indirect_direction;
            if (resampling_reservoir.sample.is_envmap_path())
                restir_resampled_indirect_direction = resampling_reservoir.sample.sample_point;
            else
                restir_resampled_indirect_direction = hippt::normalize(resampling_reservoir.sample.sample_point - closest_hit_info.inter_point);

            // Computing the BSDF throughput at the first hit
            //  - view direction: towards the camera
            //  - incident light direction: towards the sample point
            float bsdf_pdf_first_hit;
            ColorRGB32F first_hit_throughput;
            ColorRGB32F bsdf_color_first_hit = bsdf_dispatcher_eval(render_data, ray_payload.material, ray_payload.volume_state, false, view_direction, closest_hit_info.shading_normal, geometric_normal, restir_resampled_indirect_direction, bsdf_pdf_first_hit, random_number_generator, 0, resampling_reservoir.sample.incident_light_info_at_visible_point);
            if (bsdf_pdf_first_hit > 0.0f)
                first_hit_throughput = bsdf_color_first_hit * hippt::abs(hippt::dot(restir_resampled_indirect_direction, closest_hit_info.shading_normal)) * resampling_reservoir.UCW;

            if (resampling_reservoir.sample.is_envmap_path())
                camera_outgoing_radiance += path_tracing_miss_gather_envmap(render_data, first_hit_throughput, restir_resampled_indirect_direction, 1, pixel_index);
            else
            {
                if (render_data.render_settings.DEBUG_DOUBLE_BSDF_SHADING)
                {
                    // TODO create a new ray payload variable for clarity and check that the registers don't go up
                    // Loading the second hit in the ray payload:
                    ray_payload.material = resampling_reservoir.sample.sample_point_material.unpack();
                    ray_payload.volume_state = resampling_reservoir.sample.sample_point_volume_state;
                    ray_payload.bounce = 1;

                    // TODO create a new closest hit variable for clarity and check that the registers don't go up
                    // Loading the second hit in the closest hit
                    closest_hit_info.geometric_normal = resampling_reservoir.sample.sample_point_geometric_normal;
                    closest_hit_info.shading_normal = resampling_reservoir.sample.sample_point_shading_normal;
                    closest_hit_info.inter_point = resampling_reservoir.sample.sample_point;
                    closest_hit_info.primitive_index = resampling_reservoir.sample.sample_point_primitive_index;

                    // Using the same seed for direct lighting as when generating the initial candidate
                    random_number_generator.m_state.seed = resampling_reservoir.sample.direct_lighting_at_sample_point_random_seed;
                    // Taking the direct lighting at the sample point hit into account
                    ColorRGB32F direct_lighting_second_hit = estimate_direct_lighting(render_data, ray_payload, first_hit_throughput, closest_hit_info, -restir_resampled_indirect_direction, x, y, mis_reuse, random_number_generator);
                    camera_outgoing_radiance += direct_lighting_second_hit;

                    if (!resampling_reservoir.sample.incoming_radiance_to_sample_point.is_black())
                    {
                        // Computing the BSDF throughput at the second hit
                        //  - view direction: towards the first hit
                        //  - incident light direction: towards what's after the sample point (i.e. the second bounce direction)
                        float bsdf_pdf_second_hit;
                        ColorRGB32F bsdf_color_second_hit = bsdf_dispatcher_eval(render_data, ray_payload.material, ray_payload.volume_state, false, -restir_resampled_indirect_direction, resampling_reservoir.sample.sample_point_shading_normal, resampling_reservoir.sample.sample_point_geometric_normal, resampling_reservoir.sample.incident_light_direction_at_sample_point, bsdf_pdf_second_hit, random_number_generator, 1, resampling_reservoir.sample.incident_light_info_at_sample_point);
                        ColorRGB32F second_hit_throughput;
                        if (bsdf_pdf_second_hit > 0.0f)
                            second_hit_throughput = bsdf_color_second_hit * hippt::abs(hippt::dot(resampling_reservoir.sample.incident_light_direction_at_sample_point, closest_hit_info.shading_normal)) / bsdf_pdf_second_hit;

                        ColorRGB32F reconstructed = first_hit_throughput * second_hit_throughput * resampling_reservoir.sample.incoming_radiance_to_sample_point;

#ifndef __KERNELCC__
                        static std::atomic<float> max_diff = -1000000.0f;
                        float max_comp = (reconstructed - (first_hit_throughput * resampling_reservoir.sample.incoming_radiance_to_visible_point)).max_component();
                        hippt::atomic_max(&max_diff, max_comp);

                        if (max_diff == max_comp)
                            std::cout << "Max: " << max_diff << "[" << x << ", " << y << "]" << std::endl;
#endif

                        camera_outgoing_radiance += first_hit_throughput * second_hit_throughput * resampling_reservoir.sample.incoming_radiance_to_sample_point;
                    }
                }
                else
                    camera_outgoing_radiance += first_hit_throughput * resampling_reservoir.sample.incoming_radiance_to_visible_point;
            }
        }
    }

    /*if (x == render_data.render_settings.debug_x && y == render_data.render_settings.debug_y)
#ifndef __KERNELCC__
        if (render_data.render_settings.sample_number % 2 == 0)
#else
        if (render_data.render_settings.sample_number > render_data.render_settings.stop_value)
#endif
        {
            printf("Count: %d\n", hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0));
            for (int i = 0; i < 10; i++)
            {
                float value = hippt::atomic_fetch_add(&render_data.render_settings.DEBUG_SUMS[i], 0.0f);
                if (value != 0.0f)
                    printf("Average %d: %e\n", i + 1, value / hippt::atomic_fetch_add(render_data.render_settings.DEBUG_SUM_COUNT, 0));
            }

            printf("\n");
        }*/

    // Setting the 'camera_outgoing_radiance' into the ray color just for the call to 'sanity_check'
    ray_payload.ray_color = camera_outgoing_radiance;
    if (!sanity_check(render_data, ray_payload, x, y))
        return;

    if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::FINAL_RESERVOIR_UCW)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.UCW) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::TARGET_FUNCTION)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.sample.target_function) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::WEIGHT_SUM)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.weight_sum) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::M_COUNT)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.M) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else
        // Regular output
        path_tracing_accumulate_color(render_data, camera_outgoing_radiance, pixel_index);
}

#endif
