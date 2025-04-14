/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_GRID_FILL_H
#define DEVICE_KERNELS_REGIR_GRID_FILL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/ReGIR/Grid.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReGIR_Grid_Fill(HIPRTRenderData render_data, ReGIRSettings regir_settings)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Fill(HIPRTRenderData render_data, ReGIRSettings regir_settings, int linear_cell_index)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif

    ReGIRGrid& regir_grid = render_data.render_settings.regir_grid;
    if (linear_cell_index >= regir_grid.grid_resolution.x * regir_grid.grid_resolution.y * regir_grid.grid_resolution.z)
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(linear_cell_index + 1);
    else
        seed = wang_hash((linear_cell_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

    Xorshift32Generator random_number_generator(seed);


    ReGIRReservoir cell_reservoir;
	float3 cell_center = regir_grid.get_cell_center(linear_cell_index);
	for (int light_sample_index = 0; light_sample_index < regir_settings.light_samples_per_grid_cell; light_sample_index++)
	{
        LightSampleInformation light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, make_float3(0.0f, 0.0f, 0.0f), random_number_generator);
        if (light_sample.area_measure_pdf <= 0.0f)
            continue;

        float mis_weight = 1.0f / regir_settings.light_samples_per_grid_cell;
        float target_function = ReGIR_grid_fill_evaluate_target_function(render_data, cell_center, light_sample);
        float source_pdf = light_sample.area_measure_pdf;

        cell_reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, random_number_generator);
	}

    cell_reservoir.finalize_resampling();

    regir_grid.grid_buffer[linear_cell_index] = cell_reservoir;
}

#endif
