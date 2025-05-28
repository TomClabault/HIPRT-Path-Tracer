/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SUPERSAMPLING_COPY_H
#define DEVICE_KERNELS_REGIR_SUPERSAMPLING_COPY_H

#include "HostDeviceCommon/RenderData.h"

/**
 * This kernel inserts the keys of the input hash table into the output hash table
 *
 * This is used when the hash table has been resized and we need to re-insert the keys
 * of the old (smaller) hash table into the new (larger) hash table
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Supersampling_Copy(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Supersampling_Copy(HIPRTRenderData render_data, int thread_index)
#endif
{
#ifdef __KERNELCC__
    const uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

#ifdef __KERNELCC__
    if (thread_index >= *render_data.render_settings.regir_settings.hash_cell_data.grid_cells_alive_count)
#else
    if (thread_index >= render_data.render_settings.regir_settings.hash_cell_data.grid_cells_alive_count->load())
#endif
    {
        return;
    }

	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    unsigned int reservoir_index = thread_index;
    unsigned int reservoir_index_in_cell = reservoir_index % regir_settings.get_number_of_reservoirs_per_cell();
    unsigned int cell_alive_index = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell();
    // If all cells are alive, the cell index is straightforward
    //
    // Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
    // so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
    unsigned int hash_grid_cell_index = regir_settings.hash_cell_data.grid_cells_alive_list[cell_alive_index];
    unsigned int reservoir_index_in_grid = hash_grid_cell_index * regir_settings.get_number_of_reservoirs_per_cell() + reservoir_index_in_cell;

    ReGIRReservoir reservoir_to_copy;
    if (regir_settings.spatial_reuse.do_spatial_reuse)
        reservoir_to_copy = regir_settings.hash_grid.read_full_reservoir(regir_settings.spatial_output_grid, regir_settings.hash_cell_data, reservoir_index_in_grid);
    else
        reservoir_to_copy = regir_settings.hash_grid.read_full_reservoir(regir_settings.initial_reservoirs_grid, regir_settings.hash_cell_data, reservoir_index_in_grid);

	reservoir_index_in_grid += regir_settings.supersampling.supersampling_current_frame * regir_settings.get_number_of_reservoirs_per_grid();

    render_data.render_settings.regir_settings.supersampling.supersampling_grid.reservoirs.store_reservoir_opt(reservoir_index_in_grid, reservoir_to_copy);
}

#endif