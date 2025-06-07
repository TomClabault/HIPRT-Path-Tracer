/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_GRID_FILL_H
#define DEVICE_KERNELS_REGIR_GRID_FILL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/LightUtils.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"
#include "Device/includes/ReSTIR/ReGIR/VisibilityReuse.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ReGIRReservoir grid_fill(const HIPRTRenderData& render_data, const ReGIRSettings& regir_settings, int reservoir_index_in_cell, int hash_grid_cell_index,
    Xorshift32Generator& rng)
{
    ReGIRReservoir grid_fill_reservoir;

    bool reservoir_is_canonical = regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell);
    unsigned int light_sample_count = regir_settings.grid_fill.light_sample_count_per_cell_reservoir;
    unsigned int bsdf_sample_count = reservoir_is_canonical ? 0 : regir_settings.grid_fill.bsdf_sample_count_per_cell_reservoir;
 
    int cell_primitive_index = ReGIR_get_cell_primitive_index(render_data, hash_grid_cell_index);
    float3 cell_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index);
    float3 cell_normal = ReGIR_get_cell_world_shading_normal(render_data, hash_grid_cell_index);

    for (int light_sample_index = 0; light_sample_index < light_sample_count; light_sample_index++)
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);
        if (light_sample.emissive_triangle_index == -1)
            continue;

        float target_function;
        if (reservoir_is_canonical)
            // This reservoir is canonical, simple target function to keep it canonical (no visibility / cosine terms)
            target_function = ReGIR_non_shading_evaluate_target_function<false, false, false, false>(render_data, hash_grid_cell_index,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        else
            target_function = ReGIR_non_shading_evaluate_target_function<
            ReGIR_GridFillTargetFunctionVisibility,
            ReGIR_GridFillTargetFunctionCosineTerm,
            ReGIR_GridFillTargetFunctionCosineTermLightSource,
            ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation>(render_data, hash_grid_cell_index,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        
        float3 direction_to_light = light_sample.point_on_light - cell_point;
        float distance = hippt::length(direction_to_light);
        direction_to_light /= distance;

        float solid_angle_light_pdf;
        if (reservoir_is_canonical)
            solid_angle_light_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance, hippt::abs(hippt::dot(light_sample.light_source_normal, -direction_to_light)));
        else
        {
            if (ReGIR_GridFillTargetFunctionCosineTermLightSource == KERNEL_OPTION_TRUE)
                solid_angle_light_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance, compute_cosine_term_at_light_source(light_sample.light_source_normal, -direction_to_light));
            else
                solid_angle_light_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance, hippt::abs(hippt::dot(light_sample.light_source_normal, -direction_to_light)));
        }

        float bsdf_pdf;

        {
            float cell_roughness = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index);
            float cell_metallic = ReGIR_get_cell_metallic(render_data, hash_grid_cell_index);
            float cell_specular = ReGIR_get_cell_specular(render_data, hash_grid_cell_index);
            RayVolumeState empty_volume_state;
            BSDFIncidentLightInfo out_incident_light_info;
            DeviceUnpackedEffectiveMaterial approximate_material;
            approximate_material.roughness = cell_roughness;
            approximate_material.metallic = cell_metallic;
            approximate_material.specular = cell_specular;

            BSDFContext bsdf_context = BSDFContext(hippt::normalize(render_data.current_camera.position - cell_point), cell_normal, cell_normal, direction_to_light, out_incident_light_info, empty_volume_state, false, approximate_material, 0, 0, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
            bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, rng);
        }

        float mis_weight = balance_heuristic(solid_angle_light_pdf, light_sample_count, bsdf_pdf, bsdf_sample_count);

        float source_pdf = light_sample.area_measure_pdf;
        grid_fill_reservoir.stream_sample(mis_weight, target_function, source_pdf, 0, light_sample, rng);
    }

    for (int bsdf_sample_index = 0; bsdf_sample_index < bsdf_sample_count; bsdf_sample_index++)
    {
        float bsdf_sample_pdf;
        float3 sampled_bsdf_direction;

        {
            float cell_roughness = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index);
            float cell_metallic = ReGIR_get_cell_metallic(render_data, hash_grid_cell_index);
            float cell_specular = ReGIR_get_cell_specular(render_data, hash_grid_cell_index);

            RayVolumeState empty_volume_state;
            BSDFIncidentLightInfo out_incident_light_info;
            DeviceUnpackedEffectiveMaterial approximate_material;
            approximate_material.roughness = cell_roughness;
            approximate_material.metallic = cell_metallic;
            approximate_material.specular = cell_specular;

            BSDFContext bsdf_context = BSDFContext(hippt::normalize(render_data.current_camera.position - cell_point), cell_normal, cell_normal, make_float3(0, 0, 0), out_incident_light_info, empty_volume_state, false, approximate_material, 0, 0, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
            ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, rng);
        }

        if (bsdf_sample_pdf > 0.0f)
        {
            ShadowLightRayHitInfo shadow_light_ray_hit_info;

            hiprtRay new_ray;
            new_ray.origin = cell_point;
            new_ray.direction = sampled_bsdf_direction;

            // Checking that we did hit something and if we hit something,
            // it needs to be emissive
            bool intersection_found = evaluate_shadow_light_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, cell_primitive_index, 0, rng);
            if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black())
            {
                float3 bsdf_ray_inter_point = cell_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;
				float3 to_light_direction = bsdf_ray_inter_point - cell_point;
				float distance_to_light = hippt::length(to_light_direction);
				to_light_direction /= distance_to_light; // Normalizing the direction to light

                LightSampleInformation light_sample;
                if (ReGIR_GridFillTargetFunctionCosineTermLightSource == KERNEL_OPTION_TRUE)
                    light_sample.area_measure_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, distance_to_light, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                else
                    light_sample.area_measure_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, distance_to_light, hippt::abs(hippt::dot(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction)));
                //light_sample.area_measure_pdf = solid_angle_to_area_pdf(bsdf_sample_pdf, distance_to_light, compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
                light_sample.emission = shadow_light_ray_hit_info.hit_emission;
                light_sample.light_source_normal = shadow_light_ray_hit_info.hit_geometric_normal;
                light_sample.point_on_light = bsdf_ray_inter_point;
                light_sample.emissive_triangle_index = shadow_light_ray_hit_info.hit_prim_index;

                float target_function = ReGIR_non_shading_evaluate_target_function<
                    false,
                    ReGIR_GridFillTargetFunctionCosineTerm,
                    ReGIR_GridFillTargetFunctionCosineTermLightSource,
                    false>(render_data, hash_grid_cell_index,
                        light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);

                float light_sampling_PDF = pdf_of_emissive_triangle_hit_solid_angle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, shadow_light_ray_hit_info, to_light_direction);
                float mis_weight = balance_heuristic(bsdf_sample_pdf, bsdf_sample_count, light_sampling_PDF, light_sample_count);
                float bsdf_pdf_area_measure = light_sample.area_measure_pdf;

                grid_fill_reservoir.stream_sample(mis_weight, target_function, bsdf_pdf_area_measure, rng.xorshift32(), light_sample, rng);
            }
        }
    }

    return grid_fill_reservoir;
}

/** 
 * This kernel is in charge of resetting (when necessary) and filling the ReGIR grid.
 * 
 * This kernel also does the temporal reuse if enabled.
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, unsigned int number_of_cells_alive)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, int thread_index, unsigned int number_of_cells_alive)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_count = gridDim.x * blockDim.x;
#endif

    while (thread_index < regir_settings.get_number_of_reservoirs_per_cell() * number_of_cells_alive)
    {
        int reservoir_index = thread_index;
        
        ReGIRReservoir output_reservoir;

        unsigned int reservoir_index_in_cell = reservoir_index % regir_settings.get_number_of_reservoirs_per_cell();
        unsigned int cell_alive_index = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell();
        // If all cells are alive, the cell index is straightforward
        //
        // Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
        // so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
        unsigned int hash_grid_cell_index = number_of_cells_alive == regir_settings.get_total_number_of_cells_per_grid() ? cell_alive_index : regir_settings.hash_cell_data.grid_cells_alive_list[cell_alive_index];
        unsigned int reservoir_index_in_grid = hash_grid_cell_index * regir_settings.get_number_of_reservoirs_per_cell() + reservoir_index_in_cell;
        
        // Reset grid
        if (render_data.render_settings.need_to_reset)
            regir_settings.reset_reservoirs(hash_grid_cell_index, reservoir_index_in_cell);
        
        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index_in_grid + 1);
        else
            seed = wang_hash((reservoir_index_in_grid + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
        
        Xorshift32Generator random_number_generator(seed);
        
        float3 representative_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index);
        float3 normal = ReGIR_get_cell_world_shading_normal(render_data, hash_grid_cell_index);
        
        // TODO do we need this since we're only dispatching for alive grid cells anyways with the compaction?
        if (regir_settings.hash_cell_data.grid_cell_alive[hash_grid_cell_index] == 0)
        {
            // Grid cell wasn't used during shading in the last frame, let's not refill it
            
            // Storing an empty reservoir to clear the cell
            regir_settings.store_reservoir_opt(ReGIRReservoir(), representative_point, render_data.current_camera, reservoir_index_in_cell);
            
            return;
        }
        
        // Grid fill
        output_reservoir = grid_fill(render_data, regir_settings, reservoir_index_in_cell, hash_grid_cell_index, random_number_generator);
        
        // Normalizing the reservoir
        // float normalization_weight = regir_settings.grid_fill.light_sample_count_per_cell_reservoir;
        // output_reservoir.finalize_resampling(normalization_weight);
        output_reservoir.finalize_resampling(1.0f);
        
        // Discarding occluded reservoirs with visibility reuse
        if (!regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
            // Only visibility-checking non-canonical reservoirs because canonical reservoirs are never visibility-reused so that they stay canonical
            output_reservoir = visibility_reuse(render_data, output_reservoir, hash_grid_cell_index, random_number_generator);

        regir_settings.store_reservoir_opt(output_reservoir, representative_point, render_data.current_camera, reservoir_index_in_cell);

#ifndef __KERNELCC__
        // We're dispatching exactly one thread per reservoir to compute on the CPU so no need
        // for the work queue style of things that is only needed on the GPU, we can just exit here
        break;
#else
        // We need to compute the next reservoir index for the next iteration
        thread_index += thread_count;
#endif
    }
}

#endif
