/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
#define DEVICE_KERNELS_REGIR_SPATIAL_REUSE_H
 
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/RenderData.h"

 /** 
  * This kernel is in charge of the spatial reuse on the ReGIR grid.
  * 
  * Each cell reuses from random cells adjacent to it
  */
 #ifdef __KERNELCC__
 GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Spatial_Reuse(HIPRTRenderData render_data)
 #else
 GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Spatial_Reuse(HIPRTRenderData render_data, int reservoir_index)
 #endif
 {
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
    const uint32_t reservoir_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    int reservoir_index_in_cell = reservoir_index % regir_settings.grid_fill.reservoirs_count_per_grid_cell;

    if (reservoir_index >= regir_settings.get_number_of_reservoirs_per_grid())
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(reservoir_index + 1);
    else
        seed = wang_hash((reservoir_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

    Xorshift32Generator random_number_generator(seed);

    // Everyone is going to use the same RNG such that memory accesses on the spatial neighbors are coalesced
    // This is ~2x performance on a 7900XTX
    Xorshift32Generator spatial_neighbor_rng(wang_hash((render_data.render_settings.sample_number + 1) * render_data.random_number));

    ReGIRReservoir output_reservoir;

    int linear_center_cell_index = reservoir_index / regir_settings.grid_fill.reservoirs_count_per_grid_cell;
    int3 xyz_center_cell_index = regir_settings.get_xyz_cell_index_from_linear(linear_center_cell_index);

    float valid_neighbor_count = 0.0f;
    for (int neighbor_index = 0; neighbor_index < regir_settings.spatial_reuse.spatial_neighbor_reuse_count + 1; neighbor_index++)
    {
        int3 offset;
        if (neighbor_index == regir_settings.spatial_reuse.spatial_neighbor_reuse_count)
            // The last neighbor reused is the center cell
            offset = make_int3(0, 0, 0);
        else
        {
            float3 offset_float_radius_1 = make_float3(spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f, spatial_neighbor_rng() * 2.0f - 1.0f);
            float3 offset_float_radius = offset_float_radius_1 * regir_settings.spatial_reuse.spatial_reuse_radius;
            
            offset = make_int3(roundf(offset_float_radius.x), roundf(offset_float_radius.y), roundf(offset_float_radius.z));
        }

        int3 neighbor_xyz_cell_index = xyz_center_cell_index + offset;
        int neighbor_cell_linear_index_in_grid = regir_settings.get_cell_linear_index_from_xyz(neighbor_xyz_cell_index);
        if (neighbor_cell_linear_index_in_grid == -1)
            // Neighbor is outside of the grid
            continue;
        else
            valid_neighbor_count += 1.0f;

        int neighbor_reservoir_linear_index_in_grid = neighbor_cell_linear_index_in_grid * regir_settings.grid_fill.reservoirs_count_per_grid_cell + reservoir_index_in_cell;

        ReGIRReservoir neighbor_reservoir;
        if (regir_settings.temporal_reuse.do_temporal_reuse)
            // Reading from the output of the temporal reuse
            neighbor_reservoir = regir_settings.get_temporal_reservoir(neighbor_reservoir_linear_index_in_grid);
        else
            // No temporal reuse, reading from the output of the grid fill buffer
            neighbor_reservoir = regir_settings.get_grid_fill_output_reservoir(neighbor_reservoir_linear_index_in_grid);

        if (neighbor_reservoir.UCW == 0.0f)
            continue;

        float3 cell_center = regir_settings.get_cell_center(linear_center_cell_index);
        float mis_weight = 1.0f;
        float target_function_at_center = ReGIR_grid_fill_evaluate_target_function(cell_center, neighbor_reservoir.sample.emission, neighbor_reservoir.sample.point_on_light);

        output_reservoir.stream_reservoir(mis_weight, target_function_at_center, neighbor_reservoir, random_number_generator);
    }

    // Normalizing the reservoirs to 1
    output_reservoir.M = 1;
    output_reservoir.finalize_resampling(valid_neighbor_count);

    regir_settings.spatial_reuse.store_reservoir(output_reservoir, reservoir_index);
}

#endif