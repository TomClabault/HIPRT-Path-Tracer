/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_REHASH_KERNEL_H
#define DEVICE_KERNELS_REGIR_REHASH_KERNEL_H

#include "Device/includes/ReSTIR/ReGIR/Settings.h"

/**
 * This kernel inserts the keys of the input hash table into the output hash table
 *
 * This is used when the hash table has been resized and we need to re-insert the keys
 * of the old (smaller) hash table into the new (larger) hash table
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Rehash(
    HIPRTCamera current_camera,

    ReGIRHashGrid new_hash_grid,
    ReGIRHashGridSoADevice new_hash_grid_soa, ReGIRHashCellDataSoADevice new_hash_cell_data,

    ReGIRHashCellDataSoADevice old_hash_cell_data,
    unsigned int* old_grid_cells_alive_list,
    unsigned int old_cell_count, 
    
    bool primary_hit)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Rehash(
    HIPRTCamera current_camera,

    ReGIRHashGrid new_hash_grid,
    ReGIRHashGridSoADevice new_hash_grid_soa, ReGIRHashCellDataSoADevice new_hash_cell_data,

    ReGIRHashCellDataSoADevice old_hash_cell_data,
    unsigned int* old_grid_cells_alive_list, 
    unsigned int old_cell_count, 
    
    bool primary_hit,

    unsigned int cell_index
)
#endif
{
#ifdef __KERNELCC__
    const uint32_t cell_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    if (cell_index >= old_cell_count)
        return;

    unsigned int cell_alive_index = old_grid_cells_alive_list[cell_index];

    float3 world_position = old_hash_cell_data.world_points[cell_alive_index];
    float3 shading_normal = old_hash_cell_data.world_normals[cell_alive_index].unpack();
    int primitive_index = old_hash_cell_data.hit_primitive[cell_alive_index];

	DeviceUnpackedEffectiveMaterial material;
	material.roughness = old_hash_cell_data.roughness[cell_alive_index] / 255.0f;
	material.metallic = old_hash_cell_data.metallic[cell_alive_index] / 255.0f;
	material.specular = old_hash_cell_data.specular[cell_alive_index] / 255.0f;

    ReGIRSettings::insert_hash_cell_data_static(
        new_hash_grid, new_hash_grid_soa, new_hash_cell_data,
        world_position, shading_normal, current_camera, primitive_index, primary_hit, material);
}

#endif