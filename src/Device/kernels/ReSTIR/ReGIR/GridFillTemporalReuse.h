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

    for (int light_sample_index = 0; light_sample_index < regir_settings.grid_fill.sample_count_per_cell_reservoir; light_sample_index++)
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);
        if (light_sample.emissive_triangle_index == -1)
            continue;

        float target_function;
        if (reservoir_index_in_cell < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell())
            target_function = ReGIR_non_shading_evaluate_target_function<ReGIR_GridFillTargetFunctionVisibility, ReGIR_GridFillTargetFunctionCosineTerm, ReGIR_GridFillTargetFunctionCosineTermLightSource>(render_data, hash_grid_cell_index,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        else
            // This reservoir is canonical, simple target function to keep it canonical (no visibility / cosine terms)
            target_function = ReGIR_non_shading_evaluate_target_function<false, false, false>(render_data, hash_grid_cell_index, 
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);

        float source_pdf = light_sample.area_measure_pdf;
        float mis_weight = 1.0f;

        grid_fill_reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, rng);
    }

    return grid_fill_reservoir;
}

HIPRT_DEVICE ReGIRReservoir temporal_reuse(const HIPRTRenderData& render_data, const ReGIRSettings& regir_settings, int hash_grid_cell_index, int reservoir_index_in_cell, const ReGIRReservoir& current_reservoir, float& in_out_normalization_weight, Xorshift32Generator& rng)
{
    ReGIRReservoir output_reservoir = current_reservoir;

    float3 representative_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index);

    if (regir_settings.temporal_reuse.do_temporal_reuse)
    {
        // Looping over the grids of the past frames to combine the reservoirs with the current
        for (int grid_index = 0; grid_index < regir_settings.temporal_reuse.temporal_history_length; grid_index++)
        {
            if (grid_index == regir_settings.temporal_reuse.current_grid_index)
                continue;

            ReGIRReservoir past_frame_reservoir = regir_settings.get_temporal_reservoir_opt(representative_point, render_data.current_camera.position, reservoir_index_in_cell, grid_index);
            // M-capping
            past_frame_reservoir.M = hippt::min(past_frame_reservoir.M, (unsigned char)regir_settings.temporal_reuse.m_cap);

            output_reservoir.stream_reservoir(past_frame_reservoir.M, past_frame_reservoir.sample.target_function, past_frame_reservoir, rng);
            in_out_normalization_weight += past_frame_reservoir.M;
        }
    }

    return output_reservoir;
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

        float normalization_weight = regir_settings.grid_fill.sample_count_per_cell_reservoir;
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
        if (regir_settings.hash_cell_data.grid_cells_alive[hash_grid_cell_index] == 0)
        {
            // Grid cell wasn't used during shading in the last frame, let's not refill it
            
            // Storing an empty reservoir to clear the cell
            regir_settings.store_reservoir_opt(ReGIRReservoir(), representative_point, render_data.current_camera.position, reservoir_index_in_cell);
            
            return;
        }
        
        // Grid fill
        output_reservoir = grid_fill(render_data, regir_settings, reservoir_index_in_cell, hash_grid_cell_index, random_number_generator);
        
        // Temporal reuse
        output_reservoir = temporal_reuse(render_data, regir_settings, hash_grid_cell_index, reservoir_index_in_cell, output_reservoir, normalization_weight, random_number_generator);
        
        // Normalizing the reservoirs to 1
        output_reservoir.M = 1;
        output_reservoir.finalize_resampling(normalization_weight);
        
        // Discarding occluded reservoirs with visibility reuse
        if (!regir_settings.grid_fill.reservoir_index_in_cell_is_canonical(reservoir_index_in_cell))
            // Only visibility-checking non-canonical reservoirs because canonical reservoirs are never visibility-reused so that they stay canonical
            output_reservoir = visibility_reuse(render_data, output_reservoir, hash_grid_cell_index, random_number_generator);
        
        regir_settings.store_reservoir_opt(output_reservoir, representative_point, render_data.current_camera.position, reservoir_index_in_cell);

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
