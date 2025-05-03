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
  * This kernel simply copies the staging buffer is in charge of the spatial reuse on the ReGIR grid.
  * 
  * Each cell reuses from random cells adjacent to it
  */
 #ifdef __KERNELCC__
 GLOBAL_KERNEL_SIGNATURE(void) ReGIR_CellLivenessCopy(unsigned int* __restrict__ grid_cells_alive_staging_buffer, unsigned char* __restrict__ grid_cells_alive_buffer, ReGIRSettings regir_settings)
 #else
 GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_CellLivenessCopy(unsigned int* grid_cells_alive_staging_buffer, unsigned char* grid_cells_alive_buffer, ReGIRSettings regir_settings, int cell_index)
 #endif
 {
#ifdef __KERNELCC__
    const uint32_t cell_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (cell_index >= regir_settings.get_total_number_of_cells_per_grid())
        return;

    grid_cells_alive_buffer[cell_index] = grid_cells_alive_staging_buffer[cell_index] > 0 ? 1 : 0;

    // And resetting the staging buffer
    grid_cells_alive_staging_buffer[cell_index] = 0;
}

#endif