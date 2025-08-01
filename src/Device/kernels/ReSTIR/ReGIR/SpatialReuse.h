/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
#define DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
 
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/LightUtils.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "omp.h"
#endif

HIPRT_DEVICE unsigned int get_random_neighbor_hash_grid_cell_index_with_retries(HIPRTRenderData& render_data, 
    bool primary_hit, float3 point_in_center_cell, float3 center_cell_normal, float center_cell_roughness,
    Xorshift32Generator& spatial_neighbor_rng)
{
    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    unsigned int neighbor_hash_grid_cell_index_in_grid;
    bool neighbor_invalid = true;

    int retry = 0;
    while (retry < regir_settings.spatial_reuse.retries_per_neighbor && neighbor_invalid)
    {
        float3 random_neighbor = make_float3(spatial_neighbor_rng(), spatial_neighbor_rng(), spatial_neighbor_rng());

        float3 offset_float_radius_1 = random_neighbor * 2.0f - 1.0f;
        float3 offset_float_radius = offset_float_radius_1 * regir_settings.spatial_reuse.spatial_reuse_radius;
        float3 offset = make_float3(roundf(offset_float_radius.x), roundf(offset_float_radius.y), roundf(offset_float_radius.z));
        float3 point_in_neighbor_cell = point_in_center_cell + offset * regir_settings.get_cell_size(point_in_center_cell, render_data.current_camera, center_cell_roughness);
        
        neighbor_hash_grid_cell_index_in_grid = regir_settings.get_hash_grid_cell_index_from_world_pos(point_in_neighbor_cell, center_cell_normal, render_data.current_camera, center_cell_roughness, primary_hit);
        if (neighbor_hash_grid_cell_index_in_grid != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX && regir_settings.get_hash_cell_data_soa(primary_hit).grid_cell_alive[neighbor_hash_grid_cell_index_in_grid])
			// Neighbor is inside of the grid and alive, we can use it
            neighbor_invalid = false;

        retry++;
    }

    if (neighbor_invalid)
        // We couldn't find a good neighbor
        return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

    return neighbor_hash_grid_cell_index_in_grid;
}

HIPRT_DEVICE ReGIRReservoir spatial_reuse(HIPRTRenderData& render_data,
    ReGIRHashGridSoADevice& input_reservoirs,
    int reservoir_index_in_cell, int hash_grid_cell_index, 
    bool primary_hit, float3 center_cell_point, float3 center_cell_normal, float center_cell_roughness,
    Xorshift32Generator& spatial_neighbor_rng, Xorshift32Generator& random_number_generator)
{
    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
    ReGIRReservoir output_reservoir;

    int selected = 0;
    for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_count + 1; neighbor_index++)
    {
        bool is_center_cell = neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_count;

        // Getting a random neighbor and retrying a certain amount of times
        // in case the neighbor that we picked was out of the grid, in a dead cell, ...
        //
        // This is to have more chance to get a reusable neighbor --> more reuse --> less variance
        float3 random_neighbor;
        int neighbor_hash_grid_cell_index_in_grid;

        if (is_center_cell)
            neighbor_hash_grid_cell_index_in_grid = hash_grid_cell_index;
        else
        {
            neighbor_hash_grid_cell_index_in_grid = get_random_neighbor_hash_grid_cell_index_with_retries(render_data, primary_hit, center_cell_point, center_cell_normal, center_cell_roughness, spatial_neighbor_rng);
            if (neighbor_hash_grid_cell_index_in_grid == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                // Could not find a valid neighbor
                continue;
        }

        for (int neighbor_reuse = 0; neighbor_reuse < regir_settings.spatial_reuse.reuse_per_neighbor_count; neighbor_reuse++)
        {
            // Picking a random reservoir in the neighbor cell
			// If our reservoir is canonical, we pick a random canonical reservoir in the neighbor cell.
            // Same for non-canonical
            int random_reservoir_index_in_cell;
            if (regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                random_reservoir_index_in_cell = random_number_generator() * regir_settings.get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell() + regir_settings.get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();
            else
                random_reservoir_index_in_cell = random_number_generator() * regir_settings.get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell();

            ReGIRReservoir neighbor_reservoir = regir_settings.get_reservoir_from_grid_cell_index(input_reservoirs, neighbor_hash_grid_cell_index_in_grid, random_reservoir_index_in_cell);
            if (neighbor_reservoir.UCW <= 0.0f)
                continue;

            ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, neighbor_reservoir.sample.emissive_triangle_index);
            float3 point_on_light = neighbor_reservoir.sample.point_on_light;
            float3 light_source_normal = get_triangle_normal_not_normalized(render_data, neighbor_reservoir.sample.emissive_triangle_index);
            float light_source_area = hippt::length(light_source_normal) * 0.5f;
            light_source_normal /= light_source_area * 2.0f;

            float target_function_at_center;
            if (regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                target_function_at_center = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, hash_grid_cell_index, primary_hit,
                    emission, light_source_normal, point_on_light, random_number_generator);
            else
                target_function_at_center = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, hash_grid_cell_index, primary_hit,
                    emission, light_source_normal, point_on_light, random_number_generator);

            // MIS weight is 1.0f because we're going to normalize at the end instead of during the resampling
            float mis_weight = 1.0f;
            output_reservoir.stream_reservoir(mis_weight, target_function_at_center, neighbor_reservoir, random_number_generator);
        }
    }

    return output_reservoir;
}

HIPRT_DEVICE int spatial_reuse_mis_weight(HIPRTRenderData& render_data, const ReGIRReservoir& output_reservoir,
    int reservoir_index_in_cell, int hash_grid_cell_index, 
    bool primary_hit, float3 center_cell_point, float3 center_cell_normal, float center_cell_roughness,
    Xorshift32Generator& spatial_neighbor_rng, Xorshift32Generator& random_number_generator)
{
    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    // Now counting the number of neighbors that could have produced this sample for the MIS weight
    // This is 1/Z MIS weights
    int valid_neighbor_count = 0;

    if (output_reservoir.weight_sum > 0.0f)
    {
        ColorRGB32F emission = get_emission_of_triangle_from_index(render_data, output_reservoir.sample.emissive_triangle_index);
        float3 point_on_light = output_reservoir.sample.point_on_light;
        float3 light_source_normal = get_triangle_normal_not_normalized(render_data, output_reservoir.sample.emissive_triangle_index);
        float light_source_area = hippt::length(light_source_normal) * 0.5f;
        light_source_normal /= light_source_area * 2.0f;

        for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_count + 1; neighbor_index++)
        {
            bool is_center_cell = neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_count;

            int neighbor_hash_grid_cell_index_in_grid;

            if (is_center_cell)
                neighbor_hash_grid_cell_index_in_grid = hash_grid_cell_index;
            else
            {
                neighbor_hash_grid_cell_index_in_grid = get_random_neighbor_hash_grid_cell_index_with_retries(render_data, primary_hit, center_cell_point, center_cell_normal, center_cell_roughness, spatial_neighbor_rng);
                if (neighbor_hash_grid_cell_index_in_grid == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
                    // Could not find a valid neighbor
                    continue;
            }

            if (regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                // A canonical reservoir can always be produced by anyone
                valid_neighbor_count += regir_settings.spatial_reuse.reuse_per_neighbor_count;
            else
            {
                // Non-canonical sample, we need to count how many neighbors could have produced it
                if (ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_hash_grid_cell_index_in_grid, primary_hit, emission, light_source_normal, point_on_light, random_number_generator) > 0.0f)
                    valid_neighbor_count += regir_settings.spatial_reuse.reuse_per_neighbor_count;
            }
        }
    }

    return valid_neighbor_count;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE HIPRT_INLINE void spatial_reuse_pre_integration_accumulation(HIPRTRenderData& render_data, const ReGIRReservoir& output_reservoir, bool reservoir_is_canonical, unsigned int hash_grid_cell_index, bool primary_hit)
{
    if constexpr (accumulatePreIntegration)
    {
        if (render_data.render_settings.regir_settings.spatial_reuse.spatial_reuse_pass_index == render_data.render_settings.regir_settings.spatial_reuse.spatial_reuse_pass_count - 1)
        {
			// Only accumulating pre-integration factors on the last spatial reuse pass

            ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

            // Only doing the pre integration on the first sample of the frame
            float normalization;
            if (reservoir_is_canonical)
                normalization = regir_settings.get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell() * render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS;
            else
                normalization = regir_settings.get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell() * render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS;
            float integration_increment = hippt::max(0.0f, output_reservoir.sample.target_function * output_reservoir.UCW) / normalization;

            if (reservoir_is_canonical)
                hippt::atomic_fetch_add(&regir_settings.get_canonical_pre_integration_factor_buffer(primary_hit)[hash_grid_cell_index], integration_increment);
            else
                hippt::atomic_fetch_add(&regir_settings.get_non_canonical_pre_integration_factor_buffer(primary_hit)[hash_grid_cell_index], integration_increment);
        }
    }
}

 /** 
  * This kernel is in charge of the spatial reuse on the ReGIR grid.
  * 
  * Each cell reuses from random cells adjacent to it
  */
 #ifdef __KERNELCC__
 GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Spatial_Reuse(HIPRTRenderData render_data, 
     ReGIRHashGridSoADevice input_reservoirs_grid, ReGIRHashGridSoADevice output_reservoirs_grid, ReGIRHashCellDataSoADevice output_reservoirs_hash_cell_data,
     unsigned int number_of_cells_alive, bool primary_hit)
 #else
template <bool accumulatePreIntegration>
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Spatial_Reuse(HIPRTRenderData render_data,
    ReGIRHashGridSoADevice input_reservoirs_grid, ReGIRHashGridSoADevice output_reservoirs_grid, ReGIRHashCellDataSoADevice output_reservoirs_hash_cell_data,
    unsigned int number_of_cells_alive, bool primary_hit, int thread_index)
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

    while (thread_index < regir_settings.get_number_of_reservoirs_per_cell(primary_hit) * number_of_cells_alive)
    {
        int reservoir_index = thread_index;
        
        int reservoir_index_in_cell = reservoir_index % regir_settings.get_grid_fill_settings(primary_hit).get_total_reservoir_count_per_cell();
        int cell_alive_index = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell(primary_hit);
        int hash_grid_cell_index = cell_alive_index;
        if (number_of_cells_alive == regir_settings.get_total_number_of_cells_per_grid(primary_hit))
            // If all cells are alive, the cell index is straightforward
            hash_grid_cell_index = cell_alive_index;
        else
            // Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
            // so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
            hash_grid_cell_index = regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_list[cell_alive_index];
        int reservoir_index_in_grid = hash_grid_cell_index * regir_settings.get_number_of_reservoirs_per_cell(primary_hit) + reservoir_index_in_cell;
        
        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index_in_grid + 1);
        else
            seed = wang_hash((reservoir_index_in_grid + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

        Xorshift32Generator random_number_generator(seed);

        float3 center_cell_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index, primary_hit);
        float3 center_cell_normal = ReGIR_get_cell_world_shading_normal(render_data, hash_grid_cell_index, primary_hit);
        float center_cell_roughness = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index, primary_hit);

        if (regir_settings.get_hash_cell_data_soa(primary_hit).grid_cell_alive[hash_grid_cell_index] == 0)
        {
            // Grid cell wasn't used during shading in the last frame, let's not refill it
            
            // Storing an empty reservoir to clear the cell
            regir_settings.store_reservoir_custom_buffer_opt(output_reservoirs_grid,  ReGIRReservoir(), hash_grid_cell_index, reservoir_index_in_cell);
            
            return;
        }
        
        unsigned int spatial_neighbor_rng_seed;
        if (regir_settings.spatial_reuse.do_coalesced_spatial_reuse)
            // Everyone is going to use the same RNG (the RNG doesn't depend on the pixel index) 
            // such that memory accesses on the spatial neighbors are coalesced to improve performance
            spatial_neighbor_rng_seed = render_data.render_settings.freeze_random ? render_data.random_number : (render_data.render_settings.sample_number + 1) * render_data.random_number;
        else
            spatial_neighbor_rng_seed = wang_hash(seed);

        Xorshift32Generator spatial_neighbor_rng(spatial_neighbor_rng_seed);
        ReGIRReservoir output_reservoir = spatial_reuse(render_data, input_reservoirs_grid, reservoir_index_in_cell, hash_grid_cell_index, primary_hit, center_cell_point, center_cell_normal, center_cell_roughness, spatial_neighbor_rng, random_number_generator);

        spatial_neighbor_rng.m_state.seed = spatial_neighbor_rng_seed;

        int valid_neighbor_count = spatial_reuse_mis_weight(render_data, output_reservoir, 
                reservoir_index_in_cell, hash_grid_cell_index, 
                primary_hit, center_cell_point, center_cell_normal, center_cell_roughness,
                spatial_neighbor_rng, random_number_generator);

        // Normalizing the reservoirs to 1
        output_reservoir.finalize_resampling(1.0f, valid_neighbor_count);

        regir_settings.store_reservoir_custom_buffer_opt(output_reservoirs_grid, output_reservoir, hash_grid_cell_index, reservoir_index_in_cell);

#ifdef __KERNELCC__
        spatial_reuse_pre_integration_accumulation<ReGIR_GridFillSpatialReuse_AccumulatePreIntegration>(render_data, output_reservoir, regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell), hash_grid_cell_index, primary_hit);
#else
        spatial_reuse_pre_integration_accumulation<accumulatePreIntegration>(render_data, output_reservoir, regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell), hash_grid_cell_index, primary_hit);
#endif

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
