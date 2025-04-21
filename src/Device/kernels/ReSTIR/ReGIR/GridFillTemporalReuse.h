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

HIPRT_HOST_DEVICE ReGIRReservoir grid_fill(const HIPRTRenderData& render_data, const ReGIRSettings& regir_settings, int reservoir_index, int linear_cell_index,
    Xorshift32Generator& rng)
{
    ReGIRReservoir grid_fill_reservoir;

    for (int light_sample_index = 0; light_sample_index < regir_settings.grid_fill.sample_count_per_cell_reservoir; light_sample_index++)
    {
        LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);
        if (light_sample.area_measure_pdf <= 0.0f)
            continue;

        float target_function = ReGIR_grid_fill_evaluate_target_function<ReGIR_GridFillTargetFunctionVisibility>(render_data, linear_cell_index, light_sample.emission, light_sample.point_on_light, rng);
        float source_pdf = light_sample.area_measure_pdf;
        float mis_weight = 1.0f;

        grid_fill_reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, rng);
    }

    return grid_fill_reservoir;
}

HIPRT_HOST_DEVICE ReGIRReservoir temporal_reuse(const HIPRTRenderData& render_data, const ReGIRSettings& regir_settings, int reservoir_index, const ReGIRReservoir& current_reservoir, float& in_out_normalization_weight, Xorshift32Generator& rng)
{
    ReGIRReservoir output_reservoir = current_reservoir;

    if (regir_settings.temporal_reuse.do_temporal_reuse)
    {
        // Looping over the grids of the past frames to combine the reservoirs with the current
        for (int grid_index = 0; grid_index < regir_settings.temporal_reuse.temporal_history_length; grid_index++)
        {
            if (grid_index == regir_settings.temporal_reuse.current_grid_index)
                continue;

            ReGIRReservoir past_frame_reservoir = regir_settings.get_temporal_reservoir(reservoir_index, grid_index);
            // M-capping
            past_frame_reservoir.M = hippt::min(past_frame_reservoir.M, regir_settings.temporal_reuse.m_cap);

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
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, int reservoir_index)
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

    // Reset grid
    if (render_data.render_settings.need_to_reset)
        regir_settings.reset_reservoirs(reservoir_index);
    
    ReGIRReservoir output_reservoir;
    float normalization_weight = regir_settings.grid_fill.sample_count_per_cell_reservoir;
    int linear_cell_index = reservoir_index / regir_settings.grid_fill.reservoirs_count_per_grid_cell;
    float3 cell_center = regir_settings.get_cell_center_from_linear(linear_cell_index);

    // Grid fill
    output_reservoir = grid_fill(render_data, regir_settings, reservoir_index, linear_cell_index, random_number_generator);

    // Temporal reuse
    output_reservoir = temporal_reuse(render_data, regir_settings, reservoir_index, output_reservoir, normalization_weight, random_number_generator);

    // Normalizing the reservoirs to 1
    output_reservoir.M = 1;
    output_reservoir.finalize_resampling(normalization_weight);

    // Discarding occluded reservoirs with visibility reuse
    output_reservoir = visibility_reuse(render_data, output_reservoir, linear_cell_index, random_number_generator);

    regir_settings.store_reservoir(output_reservoir, reservoir_index);
}

#endif
