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

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE float compute_mesh_contribution(HIPRTRenderData& render_data, const ReGIRGridFillSurface& cell_surface, unsigned int mesh_index_for_grid_cell, bool primary_hit)
{
    float3 mesh_average_point = render_data.buffers.emissive_meshes_alias_tables.meshes_average_points[mesh_index_for_grid_cell];
    // Just wrapping the mesh power in an RGB value to be able to pass it to the 'target_function' function which doesn't take
    // just a float as argument
    ColorRGB32F total_mesh_power = ColorRGB32F(render_data.buffers.emissive_meshes_alias_tables.meshes_total_power[mesh_index_for_grid_cell]);

    Xorshift32Generator dummy_rng;
    return ReGIR_grid_fill_evaluate_target_function<
        /* visibility */ false, 
        /* cosine term at cell point */ true, 
        /* cosine term at mesh point */ false,
        ReGIR_GridFillPrimaryHitsTargetFunctionBSDF, ReGIR_GridFillSecondaryHitsTargetFunctionBSDF, 
        /* NEE++ */ true>(
            render_data, cell_surface, primary_hit, total_mesh_power, make_float3(0, 0, 0), mesh_average_point, dummy_rng);
}

/**
 * This kernel computes the contribution of all the meshes of the scene to each grid cell
 * of the hash grid
 * 
 * These contributions are then going to be used for building an alias table per each grid cell
 * which will be used to sample important emissive meshes directly, in one alias table sample
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Compute_Cells_Alias_Tables(HIPRTRenderData render_data, float* contributions_scratch_buffer, unsigned int cell_offset, bool primary_hit)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Compute_Cells_Alias_Tables(HIPRTRenderData render_data, float* contributions_scratch_buffer, unsigned int cell_offset, bool primary_hit, unsigned int thread_index)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

    ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    unsigned int thread_count_per_cell = render_data.buffers.emissive_meshes_alias_tables.alias_table_count;
    unsigned int nb_threads_dispatched = gridDim.x * blockDim.x;
    unsigned int max_thread_index = floorf(nb_threads_dispatched / (float)thread_count_per_cell) * thread_count_per_cell;
    if (thread_index >= max_thread_index)
        return;

    // Cell index in [0, number of grid cells alive]
    unsigned int cell_index = cell_offset + thread_index / render_data.buffers.emissive_meshes_alias_tables.meshes_alias_table.size;
    unsigned int hash_grid_cell_index = regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_list[cell_index];
    unsigned int mesh_index_for_grid_cell = thread_index % render_data.buffers.emissive_meshes_alias_tables.alias_table_count;

    ReGIRGridFillSurface cell_surface = ReGIR_get_cell_surface(render_data, hash_grid_cell_index, primary_hit);

    float mesh_contribution = compute_mesh_contribution(render_data, cell_surface, mesh_index_for_grid_cell, primary_hit);

    contributions_scratch_buffer[thread_index] = mesh_contribution;
}

#endif
