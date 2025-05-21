/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
#define DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
 
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"
#include "Device/includes/ReSTIR/ReGIR/VisibilityReuse.h"

#include "HostDeviceCommon/RenderData.h"

#ifndef __KERNELCC__
#include "omp.h"
#endif

HIPRT_DEVICE unsigned int get_random_neighbor_hash_grid_cell_index_with_retries(HIPRTRenderData& render_data, 
    float3 point_in_cell,
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
        float3 point_in_neighbor_cell = point_in_cell + offset * regir_settings.get_cell_size(point_in_cell, render_data.current_camera);
        
        neighbor_hash_grid_cell_index_in_grid = regir_settings.get_hash_grid_cell_index_from_world_pos_with_collision_resolve(point_in_neighbor_cell, render_data.current_camera);
        if (neighbor_hash_grid_cell_index_in_grid == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
            // Neighbor is outside of the grid
            neighbor_invalid = true;
        else if (regir_settings.hash_cell_data.grid_cells_alive[neighbor_hash_grid_cell_index_in_grid] == 0)
            // Neighbor cell isn't alive, let's not reuse it
            neighbor_invalid = true;
        else
            neighbor_invalid = false;

        retry++;
    }

    if (neighbor_invalid)
        // We couldn't find a good neighbor
        return ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY;

    return neighbor_hash_grid_cell_index_in_grid;
}

HIPRT_DEVICE ReGIRReservoir spatial_reuse(HIPRTRenderData& render_data,
    int reservoir_index_in_cell, int hash_grid_cell_index, float3 point_in_cell,
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
            neighbor_hash_grid_cell_index_in_grid = get_random_neighbor_hash_grid_cell_index_with_retries(render_data, point_in_cell, spatial_neighbor_rng);
            if (neighbor_hash_grid_cell_index_in_grid == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
                // Could not find a valid neighbor
                continue;
        }

        // Only reusing 1 reservoir in the center cell, that's why we have ternary here
        int reuse_count = is_center_cell ? (regir_settings.spatial_reuse.DEBUG_oONLY_ONE_CENTER_CELL ? 1 : regir_settings.spatial_reuse.reuse_per_neighbor_count) : regir_settings.spatial_reuse.reuse_per_neighbor_count;
        for (int neighbor_reuse = 0; neighbor_reuse < reuse_count; neighbor_reuse++)
        {
            // Picking a random reservoir in the neighbor cell
			// If our reservoir is canonical, we pick a random canonical reservoir in the neighbor cell.
            // Same for non-canonical
            int random_reservoir_index_in_cell;
            if (regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                random_reservoir_index_in_cell = random_number_generator() * regir_settings.grid_fill.get_canonical_reservoir_count_per_cell() + regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell();
            else
                random_reservoir_index_in_cell = random_number_generator() * regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell();

            float3 representative_point = ReGIR_get_cell_world_point(render_data, neighbor_hash_grid_cell_index_in_grid);

            ReGIRReservoir neighbor_reservoir;
            if (regir_settings.temporal_reuse.do_temporal_reuse)
                // Reading from the output of the temporal reuse
                neighbor_reservoir = regir_settings.get_temporal_reservoir_opt(representative_point, render_data.current_camera, random_reservoir_index_in_cell);
            else
                // No temporal reuse, reading from the output of the grid fill buffer
                neighbor_reservoir = regir_settings.get_grid_fill_output_reservoir_opt(representative_point, render_data.current_camera, random_reservoir_index_in_cell);

            if (neighbor_reservoir.UCW <= 0.0f)
                continue;

            // MIS weight is 1.0f because we're going to normalize at the end instead of during the resampling
            float mis_weight = 1.0f;
            float target_function_at_center;
            if (regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                // Never using the template visibility/cosine terms arguments for canonical reservoirs
                target_function_at_center = ReGIR_non_shading_evaluate_target_function<false, false, false>(render_data, hash_grid_cell_index, 
                    neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.light_source_normal.unpack(), neighbor_reservoir.sample.point_on_light,
                    random_number_generator);
            else
                target_function_at_center = ReGIR_non_shading_evaluate_target_function<false, true, true>(render_data, hash_grid_cell_index, 
                    neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.light_source_normal.unpack(), neighbor_reservoir.sample.point_on_light,
                    random_number_generator);

            output_reservoir.stream_reservoir(mis_weight, target_function_at_center, neighbor_reservoir, random_number_generator);
        }
    }

    return output_reservoir;
}

HIPRT_DEVICE int spatial_reuse_mis_weight(HIPRTRenderData& render_data, const ReGIRReservoir& output_reservoir,
    int reservoir_index_in_cell, int hash_grid_cell_index, float3 point_in_cell,
    Xorshift32Generator& spatial_neighbor_rng, Xorshift32Generator& random_number_generator)
{
    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    // Now counting the number of neighbors that could have produced this sample for the MIS weight
    // This is 1/Z MIS weights
    int valid_neighbor_count = 0;

    if (output_reservoir.weight_sum > 0.0f)
    {
        for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_count + 1; neighbor_index++)
        {
            bool is_center_cell = neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_count;

            int neighbor_hash_grid_cell_index_in_grid;

            if (is_center_cell)
                neighbor_hash_grid_cell_index_in_grid = hash_grid_cell_index;
            else
            {
                neighbor_hash_grid_cell_index_in_grid = get_random_neighbor_hash_grid_cell_index_with_retries(render_data, point_in_cell, spatial_neighbor_rng);
                if (neighbor_hash_grid_cell_index_in_grid == ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY)
                    // Could not find a valid neighbor
                    continue;
            }

            if (regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
            {
                // A canonical reservoir can always be produced by anyone
                if (is_center_cell)
					// Only reusing one reservoir in the center cell
                    valid_neighbor_count += regir_settings.spatial_reuse.DEBUG_oONLY_ONE_CENTER_CELL ? 1.0f : regir_settings.spatial_reuse.reuse_per_neighbor_count;
                else
                    valid_neighbor_count += regir_settings.spatial_reuse.reuse_per_neighbor_count;
            }
            else
            {
                // Non-canonical sample, we need to count how many neighbors could have produced it
                if (ReGIR_shading_can_sample_be_produced_by(render_data, output_reservoir.sample, neighbor_hash_grid_cell_index_in_grid, random_number_generator))
                {
                    if (is_center_cell)
                        valid_neighbor_count += regir_settings.spatial_reuse.DEBUG_oONLY_ONE_CENTER_CELL ? 1.0f : regir_settings.spatial_reuse.reuse_per_neighbor_count;
                    else
                        valid_neighbor_count += regir_settings.spatial_reuse.reuse_per_neighbor_count;
                }
                else
                {
                    // The sample cannot be produced

                    if (ReGIR_DoVisibilityReuse == KERNEL_OPTION_TRUE && is_center_cell && !regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                    {
                        // And it is the center cell that cannot produce it.
                        //
                        // This means that the visibility reuse pass (if it is enabled)
                        // will discard this sample so we can just discard it right now
                        // and save us the visibility reuse cost
                        //
                        // Also, only doing this on non-canonical reservoirs because we do not want to visibility-kill canonical reservoirs 
                        //
                        // We're discarding the reservoir by returning a 0 normalization weight so that
                        // the 'finalize_resampling()' computes an UCW of 0, effectively discarding the reservoir
                        return 0;
                    }
                }
            }
        }
    }

    return valid_neighbor_count;
}

 /** 
  * This kernel is in charge of the spatial reuse on the ReGIR grid.
  * 
  * Each cell reuses from random cells adjacent to it
  */
 #ifdef __KERNELCC__
 GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Spatial_Reuse(HIPRTRenderData render_data, unsigned int number_of_cells_alive)
 #else
 GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Spatial_Reuse(HIPRTRenderData render_data, int thread_index, unsigned int number_of_cells_alive)
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
        
        int reservoir_index_in_cell = reservoir_index % regir_settings.grid_fill.get_total_reservoir_count_per_cell();
        int cell_alive_index = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell();
        int hash_grid_cell_index = cell_alive_index;
        if (number_of_cells_alive == regir_settings.get_total_number_of_cells_per_grid())
            // If all cells are alive, the cell index is straightforward
            hash_grid_cell_index = cell_alive_index;
        else
            // Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
            // so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
            hash_grid_cell_index = regir_settings.hash_cell_data.grid_cells_alive_list[cell_alive_index];
        int reservoir_index_in_grid = hash_grid_cell_index * regir_settings.get_number_of_reservoirs_per_cell() + reservoir_index_in_cell;
        
        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index_in_grid + 1);
        else
            seed = wang_hash((reservoir_index_in_grid + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

        Xorshift32Generator random_number_generator(seed);

        float3 point_in_cell = regir_settings.hash_cell_data.world_points[hash_grid_cell_index];

        if (regir_settings.hash_cell_data.grid_cells_alive[hash_grid_cell_index] == 0)
        {
            // Grid cell wasn't used during shading in the last frame, let's not refill it
            
            // Storing an empty reservoir to clear the cell
            regir_settings.store_spatial_reservoir_opt(ReGIRReservoir(), point_in_cell, render_data.current_camera, reservoir_index_in_cell);
            
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
        ReGIRReservoir output_reservoir = spatial_reuse(render_data, reservoir_index_in_cell, hash_grid_cell_index, point_in_cell, spatial_neighbor_rng, random_number_generator);

        spatial_neighbor_rng.m_state.seed = spatial_neighbor_rng_seed;

        int valid_neighbor_count = spatial_reuse_mis_weight(render_data, output_reservoir, 
                reservoir_index_in_cell, hash_grid_cell_index, point_in_cell,
                spatial_neighbor_rng, random_number_generator);

        // Normalizing the reservoirs to 1
        output_reservoir.M = 1;
        output_reservoir.finalize_resampling(valid_neighbor_count);

        regir_settings.store_spatial_reservoir_opt(output_reservoir, point_in_cell, render_data.current_camera, reservoir_index_in_cell);

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
