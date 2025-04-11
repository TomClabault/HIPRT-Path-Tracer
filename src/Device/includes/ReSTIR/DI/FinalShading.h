/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_FINAL_SHADING_H
#define DEVICE_RESTIR_DI_FINAL_SHADING_H

#include "Device/includes/Envmap.h"
#include "Device/includes/ReSTIR/UtilsSpatial.h"
#include "Device/includes/ReSTIR/DI/TargetFunction.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F shade_ReSTIR_DI_reservoir(const HIPRTRenderData& render_data, 
    RayVolumeState& ray_volume_state, DeviceUnpackedEffectiveMaterial& material, int last_prim_index,
    const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
    const ReSTIRDIReservoir& reservoir, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F final_color;

    if (reservoir.UCW <= 0.0f)
        // No valid sample means no light contribution
        return ColorRGB32F(0.0f);

    float distance_to_light;

    float3 shadow_ray_direction;
    if (reservoir.sample.is_envmap_sample())
    {
        shadow_ray_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, reservoir.sample.point_on_light_source);
        distance_to_light = 1.0e35f;
    }
    else
    {
        shadow_ray_direction = reservoir.sample.point_on_light_source - shading_point;
        shadow_ray_direction = shadow_ray_direction / (distance_to_light = hippt::length(shadow_ray_direction));
    }

    bool in_shadow = false;
    if (reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED)
        in_shadow = false;
    else if (render_data.render_settings.restir_di_settings.do_final_shading_visibility)
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = shading_point;
        shadow_ray.direction = shadow_ray_direction;

        in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light, last_prim_index, /* bounce. Always 0 for ReSTIR */0, random_number_generator);
    }

    if (!in_shadow)
    {
        float bsdf_pdf;
        float cosine_at_evaluated_point;

        BSDFIncidentLightInfo incident_light_info = reservoir.sample.flags_to_BSDF_incident_light_info();
        BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, shadow_ray_direction, incident_light_info, ray_volume_state, false, material, /* bounce. Always 0 for ReSTIR DI */ 0, 0.0f, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
        ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

        cosine_at_evaluated_point = hippt::dot(shading_normal, shadow_ray_direction);
        if (reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_SAMPLED_FROM_GLASS_REFRACT_LOBE)
            // We're not allowing samples that are below the surface
            // UNLESS it's a BSDF refraction sample in which case it's valid
            // so we're restoring the cosine term to be > 0.0f so that it passes
            // the if() condition below
            cosine_at_evaluated_point = hippt::abs(cosine_at_evaluated_point);

        if (cosine_at_evaluated_point > 0.0f)
        {
            ColorRGB32F sample_emission;

            if (reservoir.sample.is_envmap_sample())
            {
                float envmap_pdf;
                sample_emission = envmap_eval(render_data, shadow_ray_direction, envmap_pdf);
            }
            else
            {
                int material_index = render_data.buffers.material_indices[reservoir.sample.emissive_triangle_index];
                sample_emission = render_data.buffers.materials_buffer.get_emission(material_index);
            }

            float area_measure_to_solid_angle_conversion;
            if (reservoir.sample.is_envmap_sample())
                area_measure_to_solid_angle_conversion = 1.0f;
            else
            {
                float3 emissive_triangle_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_index));
                area_measure_to_solid_angle_conversion = hippt::abs(hippt::dot(emissive_triangle_normal, shadow_ray_direction));
                area_measure_to_solid_angle_conversion /= hippt::square(distance_to_light);
            }

            final_color = bsdf_color * reservoir.UCW * sample_emission * cosine_at_evaluated_point * area_measure_to_solid_angle_conversion;
        }
    }

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void validate_reservoir(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir)
{
    if (reservoir.sample.is_envmap_sample() && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // Killing the reservoir if it was an envmap sample but the envmap is not used anymore
        reservoir.UCW = 0.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_light_ReSTIR_DI(const HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, int2 pixel_coords)
{
	int pixel_index = pixel_coords.x + pixel_coords.y * render_data.render_settings.render_resolution.x;

    bool decoupled_reuse_shading_enabled = render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass && ReSTIR_DI_DoSpatialNeighborsDecoupledShading == KERNEL_OPTION_TRUE;
    if (decoupled_reuse_shading_enabled)
    {
        ColorRGB32F color = render_data.render_settings.restir_di_settings.common_spatial_pass.decoupled_shading_reuse_buffer[pixel_index];

        return render_data.render_settings.restir_di_settings.common_spatial_pass.decoupled_shading_reuse_buffer[pixel_index];
    }
    
    ColorRGB32F out_color;

    // Because the spatial reuse pass runs last, the output buffer of the spatial
        // pass contains the reservoir whose sample we're going to shade
    ReSTIRDIReservoir& reservoir = render_data.render_settings.restir_di_settings.restir_output_reservoirs[pixel_index];

    // Validates the reservoir i.e. kills the reservoir if it isn't valid
    // anymore i.e. if it refers to a light that doesn't exist anymore
    validate_reservoir(render_data, reservoir);

    if (!decoupled_reuse_shading_enabled)
    {
        // No decoupled

        return shade_ReSTIR_DI_reservoir(render_data,
            ray_payload.volume_state, ray_payload.material, closest_hit_info.primitive_index,
            closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
            reservoir, random_number_generator);
    }

    float2 cos_sin_theta_rotation;
    setup_adaptive_directional_spatial_reuse<false>(const_cast<HIPRTRenderData&>(render_data), pixel_index, cos_sin_theta_rotation, random_number_generator);

    unsigned int random_neighbors_seed = render_data.render_settings.restir_di_settings.common_spatial_pass.spatial_neighbors_rng_seed;

    int valid_neighbor_count = 0;

#define DO_OVER_Z 1
#define DO_VERY_CLOSE_NEIGHBORS_OVER_M 0
#if DO_OVER_Z == 1
    if (render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass && ReSTIR_DI_DoSpatialNeighborsDecoupledShading == KERNEL_OPTION_TRUE)
    {
        Xorshift32Generator spatial_neighbors_rng(random_neighbors_seed);

        for (int i = 0; i < render_data.render_settings.restir_di_settings.common_spatial_pass.reuse_neighbor_count + 1; i++)
        {
            int neighbor_index = get_spatial_neighbor_pixel_index<false>(render_data, i, pixel_coords, cos_sin_theta_rotation, spatial_neighbors_rng);
            if (neighbor_index == -1)
                continue;
            else if (!check_neighbor_similarity_heuristics<false>(render_data, neighbor_index, pixel_index, closest_hit_info.inter_point, ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<false>(render_data, closest_hit_info.geometric_normal, closest_hit_info.shading_normal)))
                continue;

            ReSTIRDIReservoir current_sample_reservoir = render_data.render_settings.restir_di_settings.restir_output_reservoirs[neighbor_index];
            current_sample_reservoir.sample.flags &= ~ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;

            // Counting how many neighbors could have produced that sample, i.e, target_function > 0
            int valid_neighbors_for_that_sample = 0;

            Xorshift32Generator spatial_neighbors_rng_local(random_neighbors_seed);
            for (int valid_neghbor_index = 0; valid_neghbor_index < render_data.render_settings.restir_di_settings.common_spatial_pass.reuse_neighbor_count + 1; valid_neghbor_index++)
            {
                if (valid_neghbor_index == i)
                    // The neighbor itself can always produce its own samples obviously
                    valid_neighbors_for_that_sample++;
                else
                {
                    int neighbor_pixel_index = get_spatial_neighbor_pixel_index<false>(render_data, valid_neghbor_index, pixel_coords, cos_sin_theta_rotation, spatial_neighbors_rng_local);
                    if (neighbor_pixel_index == -1)
                        continue;
                    else if (!check_neighbor_similarity_heuristics<false>(render_data, neighbor_pixel_index, pixel_index, closest_hit_info.inter_point, ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<false>(render_data, closest_hit_info.geometric_normal, closest_hit_info.shading_normal)))
                        continue;

                    ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

                    // valid_neighbors_for_that_sample++;
                    valid_neighbors_for_that_sample += ReSTIR_DI_evaluate_target_function<true>(render_data, current_sample_reservoir.sample, neighbor_pixel_surface, random_number_generator) > 0;
                }
            }

            if (neighbor_index != -1)
            {
                out_color += shade_ReSTIR_DI_reservoir(render_data,
                    ray_payload.volume_state, ray_payload.material, closest_hit_info.primitive_index,
                    closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
                    current_sample_reservoir, random_number_generator) / valid_neighbors_for_that_sample;
            }
        }
    }

    valid_neighbor_count = 1;

#elif DO_VERY_CLOSE_NEIGHBORS_OVER_M == 1
    if (render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass && ReSTIR_DI_DoSpatialNeighborsDecoupledShading == KERNEL_OPTION_TRUE)
    {
        for (int i = 0; i < 5; i++)
        {
            int neighbor_index = -1;

            if (i == 4)
                // Center pixel
                neighbor_index = pixel_index;
            else
            {
                int2 offset = make_int2((i > 2 ? -1 : 1) * (i + 1) & 1, (i > 2 ? -1 : 1) * (i + 1) & 2);
                offset *= 2;
                int2 neighbor_pixel_coords = pixel_coords + offset;
                if (neighbor_pixel_coords.x >= render_data.render_settings.render_resolution.x || neighbor_pixel_coords.x < 0 ||
                    neighbor_pixel_coords.y >= render_data.render_settings.render_resolution.y || neighbor_pixel_coords.y < 0)
                    neighbor_index = -1;
                else
                    neighbor_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * render_data.render_settings.render_resolution.x;
            }

            if (neighbor_index != -1)
            {
                ReSTIRDIReservoir reservoir_to_shade = render_data.render_settings.restir_di_settings.restir_output_reservoirs[neighbor_index];

                out_color += shade_ReSTIR_DI_reservoir(render_data,
                    ray_payload.volume_state, ray_payload.material, closest_hit_info.primitive_index,
                    closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
                    reservoir_to_shade, random_number_generator);
                valid_neighbor_count++;
            }
        }
    }
#endif

    return out_color / valid_neighbor_count;
}

#endif
