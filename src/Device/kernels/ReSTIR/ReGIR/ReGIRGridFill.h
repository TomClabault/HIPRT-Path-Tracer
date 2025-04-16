/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_GRID_FILL_H
#define DEVICE_KERNELS_REGIR_GRID_FILL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReGIR_Grid_Fill(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Fill(HIPRTRenderData render_data, int reservoir_index)
#endif
{
    // if (do_spatial_reuse)
    // {
    //     if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
    //         // No initial candidates to sample since no lights
    //         return;

    // #ifdef __KERNELCC__
    //     const uint32_t reservoir_index = blockIdx.x * blockDim.x + threadIdx.x;
    // #endif

    //     ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
    //     if (reservoir_index >= regir_settings.get_number_of_reservoirs_per_grid())
    //         return;

    //     if (render_data.render_settings.need_to_reset && regir_settings.do_temporal_reuse)
    //         // Resetting all the reservoirs
    //         for (int grid_index = 0; grid_index < regir_settings.temporal_history_length; grid_index++)
    //             regir_settings.store_reservoir(ReGIRReservoir(), reservoir_index, grid_index);

    //     unsigned int seed;
    //     if (render_data.render_settings.freeze_random)
    //         seed = wang_hash(reservoir_index + 1);
    //     else
    //         seed = wang_hash((reservoir_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

    //     Xorshift32Generator random_number_generator(seed);


    //     ReGIRReservoir output_reservoir;
    //     int linear_cell_index = reservoir_index / regir_settings.reservoirs_count_per_grid_cell;
    //     int3 xyz_cell_index regir_settings.get_xyz_cell_index_from_linear(linear_cell_index);
    //     float3 cell_center = regir_settings.get_cell_center(linear_cell_index);

    //     float normalization_weight = regir_settings.sample_count_per_cell_reservoir;
    //     for (int light_sample_index = 0; light_sample_index < regir_settings.sample_count_per_cell_reservoir; light_sample_index++)
    //     {
    //         LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, random_number_generator);
    //         if (light_sample.area_measure_pdf <= 0.0f)
    //             continue;

    //         float target_function = ReGIR_grid_fill_evaluate_target_function(render_data, cell_center, light_sample);
    //         float source_pdf = light_sample.area_measure_pdf;
    //         float mis_weight = 1.0f;

    //         output_reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, random_number_generator);
    //     }

    //     if (regir_settings.do_temporal_reuse)
    //     {
    //         // Looping over the grids of the past frames to combine the reservoirs with the current
    //         for (int grid_index = 0; grid_index < regir_settings.temporal_history_length; grid_index++)
    //         {
    //             if (grid_index == regir_settings.current_grid_index)
    //                 continue;

    //             ReGIRReservoir past_frame_reservoir = regir_settings.get_reservoir(reservoir_index, grid_index);
    //             // M-capping
    //             past_frame_reservoir.M = hippt::min(past_frame_reservoir.M, regir_settings.m_cap);

    //             output_reservoir.stream_reservoir(past_frame_reservoir.M, past_frame_reservoir.sample.target_function, past_frame_reservoir, random_number_generator);
    //             normalization_weight += past_frame_reservoir.M;
    //         }
    //     }

    //     // Normalizing the reservoirs to 1
    //     output_reservoir.M = 1;
    //     output_reservoir.finalize_resampling(normalization_weight);

    //     regir_settings.store_reservoir(output_reservoir, reservoir_index);
    // }
    // else
    {
            if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
            // No initial candidates to sample since no lights
            return;

    #ifdef __KERNELCC__
        const uint32_t reservoir_index = blockIdx.x * blockDim.x + threadIdx.x;
    #endif

        ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
        if (reservoir_index >= regir_settings.get_number_of_reservoirs_per_grid())
            return;

        if (render_data.render_settings.need_to_reset && regir_settings.do_temporal_reuse)
            // Resetting all the reservoirs
            for (int grid_index = 0; grid_index < regir_settings.temporal_history_length; grid_index++)
                regir_settings.store_reservoir(ReGIRReservoir(), reservoir_index, grid_index);

        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index + 1);
        else
            seed = wang_hash((reservoir_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

        Xorshift32Generator random_number_generator(seed);


        ReGIRReservoir output_reservoir;
        int linear_cell_index = reservoir_index / regir_settings.reservoirs_count_per_grid_cell;
        float3 cell_center = regir_settings.get_cell_center(linear_cell_index);

        float normalization_weight = regir_settings.sample_count_per_cell_reservoir;
        for (int light_sample_index = 0; light_sample_index < regir_settings.sample_count_per_cell_reservoir; light_sample_index++)
        {
            LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, random_number_generator);
            if (light_sample.area_measure_pdf <= 0.0f)
                continue;

            float target_function = ReGIR_grid_fill_evaluate_target_function(render_data, cell_center, light_sample);
            float source_pdf = light_sample.area_measure_pdf;
            float mis_weight = 1.0f;

            output_reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, random_number_generator);
        }

        if (regir_settings.do_temporal_reuse)
        {
            // Looping over the grids of the past frames to combine the reservoirs with the current
            for (int grid_index = 0; grid_index < regir_settings.temporal_history_length; grid_index++)
            {
                if (grid_index == regir_settings.current_grid_index)
                    continue;

                ReGIRReservoir past_frame_reservoir = regir_settings.get_reservoir(reservoir_index, grid_index);
                // M-capping
                past_frame_reservoir.M = hippt::min(past_frame_reservoir.M, regir_settings.m_cap);

                output_reservoir.stream_reservoir(past_frame_reservoir.M, past_frame_reservoir.sample.target_function, past_frame_reservoir, random_number_generator);
                normalization_weight += past_frame_reservoir.M;
            }
        }

        // Normalizing the reservoirs to 1
        output_reservoir.M = 1;
        output_reservoir.finalize_resampling(normalization_weight);

        regir_settings.store_reservoir(output_reservoir, reservoir_index);
    }
}

#endif
