/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_GI_SHADE_RESERVOIR_H
#define DEVICE_INCLUDES_RESTIR_GI_SHADE_RESERVOIR_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/Lights.h"
#include "Device/includes/MISBSDFRayReuse.h"
#include "Device/includes/PathTracing.h"

#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

HIPRT_HOST_DEVICE ColorRGB32F shade_ReSTIR_GI_reservoir(HIPRTRenderData& render_data,
    ReSTIRGIReservoir& reservoir_to_shade,
    const float3& view_direction, RayPayload& ray_payload, HitInfo& closest_hit_info,
    int x, int y, Xorshift32Generator& random_number_generator)
{
    int pixel_index = x + render_data.render_settings.render_resolution.x * y;

    // Dummy mis_reuse variable
    MISBSDFRayReuse mis_reuse;
    ColorRGB32F camera_outgoing_radiance;
    if (render_data.render_settings.enable_direct)
        // Adding the direct lighting contribution at the first hit in the direction of the camera
        camera_outgoing_radiance += estimate_direct_lighting(render_data, ray_payload, closest_hit_info, view_direction, x, y, mis_reuse, random_number_generator);

    if (render_data.render_settings.nb_bounces > 0)
    {
        // Only doing the ReSTIR GI stuff if we have more than 1 bounce

        if (reservoir_to_shade.UCW > 0.0f)
        {
            // Only doing the shading if we do actually have a sample

            float3 geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();

            float3 restir_resampled_indirect_direction;
            if (reservoir_to_shade.sample.is_envmap_path())
                restir_resampled_indirect_direction = reservoir_to_shade.sample.sample_point;
            else
                restir_resampled_indirect_direction = hippt::normalize(reservoir_to_shade.sample.sample_point - closest_hit_info.inter_point);


            // Computing the BSDF throughput at the first hit
            //  - view direction: towards the camera
            //  - incident light direction: towards the sample point
            float bsdf_pdf_first_hit;
            BSDFContext bsdf_first_hit_context(view_direction, closest_hit_info.shading_normal, geometric_normal, restir_resampled_indirect_direction, reservoir_to_shade.sample.incident_light_info_at_visible_point, ray_payload.volume_state, false, ray_payload.material, 0, 0.0f);
            ColorRGB32F bsdf_color_first_hit = bsdf_dispatcher_eval(render_data, bsdf_first_hit_context, bsdf_pdf_first_hit, random_number_generator);

            ColorRGB32F first_hit_throughput;
            if (bsdf_pdf_first_hit > 0.0f)
                first_hit_throughput = bsdf_color_first_hit * hippt::abs(hippt::dot(restir_resampled_indirect_direction, closest_hit_info.shading_normal)) * reservoir_to_shade.UCW;

            if (reservoir_to_shade.sample.is_envmap_path())
                camera_outgoing_radiance += path_tracing_miss_gather_envmap(render_data, first_hit_throughput, restir_resampled_indirect_direction, 1, pixel_index);
            else
            {
                if (render_data.render_settings.DEBUG_DOUBLE_BSDF_SHADING)
                {
                    // TODO create a new ray payload variable for clarity and check that the registers don't go up
                    // Loading the second hit in the ray payload:
                    ray_payload.material = reservoir_to_shade.sample.sample_point_material.unpack();
                    ray_payload.volume_state = reservoir_to_shade.sample.sample_point_volume_state;
                    ray_payload.bounce = 1;
                    ray_payload.accumulate_roughness(reservoir_to_shade.sample.incident_light_info_at_visible_point);

                    // TODO create a new closest hit variable for clarity and check that the registers don't go up
                    // Loading the second hit in the closest hit
                    closest_hit_info.geometric_normal = reservoir_to_shade.sample.sample_point_geometric_normal;
                    closest_hit_info.shading_normal = reservoir_to_shade.sample.sample_point_shading_normal;
                    closest_hit_info.inter_point = reservoir_to_shade.sample.sample_point;
                    closest_hit_info.primitive_index = reservoir_to_shade.sample.sample_point_primitive_index;

                    // Using the same seed for direct lighting as when generating the initial candidate
                    random_number_generator.m_state.seed = reservoir_to_shade.sample.direct_lighting_at_sample_point_random_seed;

                    // Taking the direct lighting at the sample point hit into account
                    ColorRGB32F direct_lighting_second_hit = estimate_direct_lighting(render_data, ray_payload, first_hit_throughput, closest_hit_info, -restir_resampled_indirect_direction, x, y, mis_reuse, random_number_generator);
                    camera_outgoing_radiance += direct_lighting_second_hit;

                    if (!reservoir_to_shade.sample.incoming_radiance_to_sample_point.is_black())
                    {
                        // Computing the BSDF throughput at the second hit
                        //  - view direction: towards the first hit
                        //  - incident light direction: towards what's after the sample point (i.e. the second bounce direction)
                        float bsdf_pdf_second_hit;
                        BSDFContext bsdf_second_hit_context(hippt::normalize(-restir_resampled_indirect_direction), hippt::normalize(reservoir_to_shade.sample.sample_point_shading_normal), hippt::normalize(reservoir_to_shade.sample.sample_point_geometric_normal), hippt::normalize(reservoir_to_shade.sample.incident_light_direction_at_sample_point), reservoir_to_shade.sample.incident_light_info_at_sample_point, ray_payload.volume_state, false, ray_payload.material, 1, 0.0f);
                        ColorRGB32F bsdf_color_second_hit = bsdf_dispatcher_eval(render_data, bsdf_second_hit_context, bsdf_pdf_second_hit, random_number_generator);
                        ColorRGB32F second_hit_throughput;
                        if (bsdf_pdf_second_hit > 0.0f)
                            second_hit_throughput = bsdf_color_second_hit * hippt::abs(hippt::dot(reservoir_to_shade.sample.incident_light_direction_at_sample_point, reservoir_to_shade.sample.sample_point_shading_normal)) / bsdf_pdf_second_hit;

                        camera_outgoing_radiance += first_hit_throughput * second_hit_throughput * reservoir_to_shade.sample.incoming_radiance_to_sample_point;
                    }
                }
                else
                    camera_outgoing_radiance += first_hit_throughput * reservoir_to_shade.sample.incoming_radiance_to_visible_point;
            }
        }
    }

    if (!sanity_check(render_data, camera_outgoing_radiance, x, y))
        return ColorRGB32F(0.0f);

    if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::FINAL_RESERVOIR_UCW)
        return ColorRGB32F(reservoir_to_shade.UCW) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor;
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::TARGET_FUNCTION)
        return ColorRGB32F(reservoir_to_shade.sample.target_function) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor;
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::WEIGHT_SUM)
        return ColorRGB32F(reservoir_to_shade.weight_sum) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor;
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::M_COUNT)
        return ColorRGB32F(reservoir_to_shade.M) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor;
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::PER_PIXEL_REUSE_RADIUS && render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius != nullptr)
    {
        float radius_percentage = (render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius[pixel_index] / (float)render_data.render_settings.restir_gi_settings.common_spatial_pass.reuse_radius);
        ColorRGB32F debug_color = hippt::lerp(ColorRGB32F(2.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 2.0f, 0.0f), radius_percentage);

        return debug_color;
    }
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::PER_PIXEL_VALID_DIRECTIONS_PERCENTAGE && render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius != nullptr)
    {
        unsigned char accepted_directions = hippt::popc(ReSTIRSettingsHelper::get_spatial_reuse_direction_mask_ull<true>(render_data, pixel_index));
        float accepted_percentage = accepted_directions / 32.0f;
        ColorRGB32F debug_color = hippt::lerp(ColorRGB32F(2.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 2.0f, 0.0f), accepted_percentage);

        return debug_color;
    }
    else
        // Regular output
        return camera_outgoing_radiance;
}

HIPRT_HOST_DEVICE ColorRGB32F shade_ReSTIR_GI_reservoir(HIPRTRenderData& render_data,
    ReSTIRSurface& shading_surface, ReSTIRGIReservoir& reservoir_to_shade,
    int x, int y, Xorshift32Generator& random_number_generator)
{
    RayPayload ray_payload;
    ray_payload.volume_state = shading_surface.ray_volume_state; 
    ray_payload.material = shading_surface.material;

    HitInfo hit_info;
    hit_info.geometric_normal = shading_surface.geometric_normal;
    hit_info.shading_normal = shading_surface.shading_normal;
    hit_info.inter_point = shading_surface.shading_point;
    hit_info.primitive_index = shading_surface.last_hit_primitive_index;

    return shade_ReSTIR_GI_reservoir(render_data, reservoir_to_shade, shading_surface.view_direction, ray_payload, hit_info, x, y, random_number_generator);
}

#endif
