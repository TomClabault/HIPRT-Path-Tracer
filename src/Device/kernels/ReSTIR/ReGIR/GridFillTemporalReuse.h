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

HIPRT_DEVICE ReGIRReservoir grid_fill(const HIPRTRenderData& render_data, const ReGIRSettings& regir_settings,
    int reservoir_index_in_cell, const ReGIRGridFillSurface& surface, bool primary_hit,
    Xorshift32Generator& rng)
{
    ReGIRReservoir grid_fill_reservoir;

    bool reservoir_is_canonical = regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell);

    for (int light_sample_index = 0; light_sample_index < regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir; light_sample_index++)
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);
        if (light_sample.emissive_triangle_index == -1)
            continue;

        float target_function;
        if (reservoir_is_canonical)
            // This reservoir is canonical, simple target function to keep it canonical (no visibility / cosine terms)
            target_function = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, surface,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        else
            target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, surface,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        
        float mis_weight = 1.0f / regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir;
        float source_pdf = light_sample.area_measure_pdf;

        grid_fill_reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, rng);
    }

    return grid_fill_reservoir;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE void grid_fill_pre_integration_accumulation(HIPRTRenderData& render_data, const ReGIRReservoir& output_reservoir, bool reservoir_is_canonical, unsigned int hash_grid_cell_index, bool primary_hit)
{
    if constexpr (accumulatePreIntegration)
    {
        ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

        // Only doing the pre integration on the first sample of the frame
        // and if we don't have spatial reuse. If we have the spatial reuse, it's
        // the spatial reuse pass that will do the pre integration accumulation
        if (!regir_settings.spatial_reuse.do_spatial_reuse)
        {
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
 * This kernel is in charge of resetting (when necessary) and filling the ReGIR grid.
 * 
 * This kernel also does the temporal reuse if enabled.
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, unsigned int number_of_cells_alive, bool primary_hit)
#else
template <bool accumulatePreIntegration>
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, int thread_index, unsigned int number_of_cells_alive, bool primary_hit)
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
        
        unsigned int reservoir_index_in_cell = reservoir_index % regir_settings.get_number_of_reservoirs_per_cell(primary_hit);
        unsigned int cell_alive_index = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell(primary_hit);
        // If all cells are alive, the cell index is straightforward
        //
        // Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
        // so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
        unsigned int hash_grid_cell_index = regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_list[cell_alive_index];
        unsigned int reservoir_index_in_grid = hash_grid_cell_index * regir_settings.get_number_of_reservoirs_per_cell(primary_hit) + reservoir_index_in_cell;
        
        // Reset grid
        if (render_data.render_settings.need_to_reset)
            regir_settings.reset_reservoirs(hash_grid_cell_index, reservoir_index_in_cell, primary_hit);
        
        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index_in_grid + 1);
        else
            seed = wang_hash((reservoir_index_in_grid + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
        
        Xorshift32Generator random_number_generator(seed);
        ReGIRReservoir output_reservoir;

        ReGIRGridFillSurface cell_surface = ReGIR_get_cell_surface(render_data, hash_grid_cell_index, primary_hit);

        // TODO do we need this since we're only dispatching for alive grid cells anyways with the compaction?
        if (regir_settings.get_hash_cell_data_soa(primary_hit).grid_cell_alive[hash_grid_cell_index] == 0)
        {
            // Grid cell wasn't used during shading in the last frame, let's not refill it
            
            // Storing an empty reservoir to clear the cell
            regir_settings.store_reservoir_opt(ReGIRReservoir(), hash_grid_cell_index, primary_hit, reservoir_index_in_cell);
            
            return;
        }
        
        // Grid fill
        output_reservoir = grid_fill(render_data, regir_settings, reservoir_index_in_cell, cell_surface, primary_hit, random_number_generator);
        
        // Normalizing the reservoir
        output_reservoir.finalize_resampling(1.0f, 1.0f);
        
        // Discarding occluded reservoirs with visibility reuse
        if (!regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
            // Only visibility-checking non-canonical reservoirs because canonical reservoirs are never visibility-reused so that they stay canonical
            output_reservoir = visibility_reuse(render_data, output_reservoir, hash_grid_cell_index, random_number_generator);

        // TODO store reservoir recomputes the hash grid cell index but we already have it
        regir_settings.store_reservoir_opt(output_reservoir, hash_grid_cell_index, primary_hit, reservoir_index_in_cell);

#ifdef __KERNELCC__
        grid_fill_pre_integration_accumulation<ReGIR_GridFillSpatialReuse_AccumulatePreIntegration>(render_data, output_reservoir, regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell), hash_grid_cell_index, primary_hit);
#else
        grid_fill_pre_integration_accumulation<accumulatePreIntegration>(render_data, output_reservoir, regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell), hash_grid_cell_index, primary_hit);
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
