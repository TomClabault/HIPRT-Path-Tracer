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

 /** 
  * This kernel is in charge of the spatial reuse on the ReGIR grid.
  * 
  * Each cell reuses from random cells adjacent to it
  */
 #ifdef __KERNELCC__
 GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Spatial_Reuse(HIPRTRenderData render_data)
 #else
 GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Spatial_Reuse(HIPRTRenderData render_data, int thread_index)
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

    while (thread_index < regir_settings.get_number_of_reservoirs_per_cell() * regir_settings.shading.grid_cells_alive_count)
    {
        int reservoir_index = thread_index;

        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index + 1);
        else
            seed = wang_hash((reservoir_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

        Xorshift32Generator random_number_generator(seed);

        
        ReGIRReservoir output_reservoir;
        
        int reservoir_index_in_cell = reservoir_index % regir_settings.grid_fill.get_total_reservoir_count_per_cell();
        int cell_alive_index = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell();
        int linear_center_cell_index = cell_alive_index;
        if (regir_settings.shading.grid_cells_alive_count == regir_settings.get_total_number_of_cells())
        // If all cells are alive, the cell index is straightforward
        linear_center_cell_index = cell_alive_index;
        else
        // Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
        // so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
        linear_center_cell_index = regir_settings.shading.grid_cells_alive_list[cell_alive_index];
        int reservoir_index_in_grid = linear_center_cell_index * regir_settings.get_number_of_reservoirs_per_cell() + reservoir_index_in_cell;
        
        if (regir_settings.shading.grid_cells_alive[linear_center_cell_index] == 0)
        {
            // Grid cell wasn't used during shading in the last frame, let's not refill it
            
            // Storing an empty reservoir to clear the cell
            regir_settings.spatial_reuse.store_reservoir_opt(ReGIRReservoir(), reservoir_index_in_grid);
            
            return;
        }
        
        int3 xyz_center_cell_index = regir_settings.get_xyz_cell_index_from_linear(linear_center_cell_index);
        
        unsigned int spatial_neighbor_rng_seed;
        if (regir_settings.spatial_reuse.do_coalesced_spatial_reuse)
            // Everyone is going to use the same RNG (the RNG doesn't depend on the pixel index) 
            // such that memory accesses on the spatial neighbors are coalesced to improve performance
            spatial_neighbor_rng_seed = (render_data.render_settings.sample_number + 1) * render_data.random_number;
        else
            spatial_neighbor_rng_seed = wang_hash(seed);
        Xorshift32Generator spatial_neighbor_rng(spatial_neighbor_rng_seed);
        float3 random_neighbor = make_float3(spatial_neighbor_rng(), spatial_neighbor_rng(), spatial_neighbor_rng());

        int selected = 0;
        for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_reuse_count + 1; neighbor_index++)
        {
            int3 offset;
            if (neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_reuse_count)
                // The last neighbor reused is the center cell
                offset = make_int3(0, 0, 0);
            else
            {
                float3 offset_float_radius_1;

                if (regir_settings.DEBUG_DO_FIXED_SPATIAL_REUSE)
                    offset_float_radius_1 = random_neighbor * 2.0f - 1.0f;
                else
                    offset_float_radius_1 = make_float3(spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f);
                float3 offset_float_radius = offset_float_radius_1 * regir_settings.spatial_reuse.spatial_reuse_radius;

                offset = make_int3(roundf(offset_float_radius.x), roundf(offset_float_radius.y), roundf(offset_float_radius.z));
            }

            int3 neighbor_xyz_cell_index = xyz_center_cell_index + offset;
            int neighbor_linear_cell_index_in_grid = regir_settings.get_linear_cell_index_from_xyz(neighbor_xyz_cell_index);
            if (neighbor_linear_cell_index_in_grid == -1)
                // Neighbor is outside of the grid
                continue;
            else if (regir_settings.shading.grid_cells_alive[neighbor_linear_cell_index_in_grid] == 0)
                // Neighbor cell isn't alive, let's not reuse it
                continue;

            // Picking the same reservoir cell-index in the a neighbor cell
            int random_reservoir_index_in_cell;
            if (regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                random_reservoir_index_in_cell = random_number_generator() * regir_settings.grid_fill.get_canonical_reservoir_count_per_cell() + regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell();
            else
                random_reservoir_index_in_cell = random_number_generator() * regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell();

            int neighbor_reservoir_linear_index_in_grid;
            if (regir_settings.DEBUG_DO_FIXED_SPATIAL_REUSE)
                neighbor_reservoir_linear_index_in_grid = neighbor_linear_cell_index_in_grid * regir_settings.grid_fill.get_total_reservoir_count_per_cell() + random_reservoir_index_in_cell;
            else
                neighbor_reservoir_linear_index_in_grid = neighbor_linear_cell_index_in_grid * regir_settings.grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;

            ReGIRReservoir neighbor_reservoir;
            if (regir_settings.temporal_reuse.do_temporal_reuse)
                // Reading from the output of the temporal reuse
                neighbor_reservoir = regir_settings.get_temporal_reservoir_opt(neighbor_reservoir_linear_index_in_grid);
            else
                // No temporal reuse, reading from the output of the grid fill buffer
                neighbor_reservoir = regir_settings.get_grid_fill_output_reservoir_opt(neighbor_reservoir_linear_index_in_grid);

            if (neighbor_reservoir.UCW <= 0.0f)
                continue;

            // MIS weight is 1.0f because we're going to normalize at the end
            float mis_weight = 1.0f;
            float target_function_at_center;
            if (regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                // Never using the template visibility/cosine terms arguments for canonical reservoirs
                target_function_at_center = ReGIR_non_shading_evaluate_target_function<false, false>(render_data, linear_center_cell_index, neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.point_on_light, random_number_generator);
            else
                target_function_at_center = ReGIR_non_shading_evaluate_target_function<false, true>(render_data, linear_center_cell_index, neighbor_reservoir.sample.emission.unpack(), neighbor_reservoir.sample.point_on_light, random_number_generator);

            output_reservoir.stream_reservoir(mis_weight, target_function_at_center, neighbor_reservoir, random_number_generator);
        }

        spatial_neighbor_rng.m_state.seed = spatial_neighbor_rng_seed;

        // Now counting the number of neighbors that could have produced this sample for the MIS weight
        // This is 1/Z MIS weights
        float valid_neighbor_count = 0.0f;
        if (output_reservoir.weight_sum > 0.0f)
        {
            for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_reuse_count + 1; neighbor_index++)
            {
                int3 offset;
                if (neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_reuse_count)
                    // The last neighbor reused is the center cell
                    offset = make_int3(0, 0, 0);
                else
                {
                    float3 offset_float_radius_1;

                    if (regir_settings.DEBUG_DO_FIXED_SPATIAL_REUSE)
                        offset_float_radius_1 = random_neighbor * 2.0f - 1.0f;
                    else
                        offset_float_radius_1 = make_float3(spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f);
                    float3 offset_float_radius = offset_float_radius_1 * regir_settings.spatial_reuse.spatial_reuse_radius;

                    offset = make_int3(roundf(offset_float_radius.x), roundf(offset_float_radius.y), roundf(offset_float_radius.z));
                }

                int3 neighbor_xyz_cell_index = xyz_center_cell_index + offset;
                int neighbor_linear_cell_index_in_grid = regir_settings.get_linear_cell_index_from_xyz(neighbor_xyz_cell_index);
                if (neighbor_linear_cell_index_in_grid == -1)
                    // Neighbor is outside of the grid
                    continue;
                else if (regir_settings.shading.grid_cells_alive[neighbor_linear_cell_index_in_grid] == 0)
                    // Neighbor cell isn't alive, let's not reuse it
                    continue;

                int neighbor_reservoir_linear_index_in_grid = neighbor_linear_cell_index_in_grid * regir_settings.grid_fill.get_total_reservoir_count_per_cell() + reservoir_index_in_cell;

                if (regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
                    // A canonical reservoir can always be produced by anyone
                    valid_neighbor_count += 1.0f;
                else
                {
                    // Non-canonical sample, we need to count how many neighbors could have produced it
                    if (ReGIR_shading_can_sample_be_produced_by(render_data, output_reservoir.sample, neighbor_linear_cell_index_in_grid, random_number_generator))
                    {
                        if (neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_reuse_count && regir_settings.DEBUG_DO_FIXED_SPATIAL_REUSE)
                        {
                            valid_neighbor_count += regir_settings.spatial_reuse.spatial_neighbor_reuse_count;

                            neighbor_index = regir_settings.spatial_reuse.spatial_neighbor_reuse_count - 1;
                        }
                        else
                            valid_neighbor_count += 1.0f;
                    }
                }
            }
        }

        // Normalizing the reservoirs to 1
        output_reservoir.M = 1;
        output_reservoir.finalize_resampling(valid_neighbor_count);

        if (reservoir_index_in_cell < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell())
            // Only visibility-checking non-canonical reservoirs because canonical reservoirs are never visibility-reused so that they stay canonical
            //
            // This visibility check of the reservoirs is needed such that the shading at path tracing time
            // can properly assess whether a given cell could have produced a given sample or not
            output_reservoir = visibility_reuse(render_data, output_reservoir, linear_center_cell_index, random_number_generator);

        regir_settings.spatial_reuse.store_reservoir_opt(output_reservoir, reservoir_index_in_grid);

#ifdef  __KERNELCC__
        // We need to compute the next reservoir index for the next iteration
        thread_index += thread_count;
#endif // ! __KERNELCC__
    }
}

#endif
