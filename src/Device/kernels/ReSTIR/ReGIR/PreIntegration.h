/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_REGIR_PRE_INTEGRATION_H
#define KERNELS_REGIR_PRE_INTEGRATION_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReGIR_Pre_integration(HIPRTRenderData render_data, unsigned int number_of_cells_alive)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Pre_integration(HIPRTRenderData render_data, unsigned int number_of_cells_alive, int thread_index)
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

    while (thread_index < number_of_cells_alive)
    {
        int cell_alive_index = thread_index;

        unsigned int hash_grid_cell_index = number_of_cells_alive == regir_settings.get_total_number_of_cells_per_grid() ? cell_alive_index : regir_settings.hash_cell_data.grid_cells_alive_list[cell_alive_index];

        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(hash_grid_cell_index + 1);
        else
            seed = wang_hash((hash_grid_cell_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);

        Xorshift32Generator random_number_generator(seed);

        int primitive_index = ReGIR_get_cell_primitive_index(render_data, hash_grid_cell_index);
        float3 representative_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index);
        float3 normal = ReGIR_get_cell_world_shading_normal(render_data, hash_grid_cell_index);
        float roughness = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index);


		// This kernel always uses a Lambertian BRDF where the view direction is not used so it can be set to zero
        float3 view_direction = make_float3(0.0f, 0.0f, 0.0f);
		RayPayload ray_payload;
		ray_payload.material.roughness = roughness;

        float non_canonical_cell_integration_sum = 0.0f;
        for (int i = 0; i < regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell() / 2; i++)
        {
            bool invalid_sample = false;
            ReGIRReservoir non_canonical_reservoir = regir_settings.get_random_cell_non_canonical_reservoir(representative_point, render_data.current_camera, roughness, random_number_generator, &invalid_sample);
            if (invalid_sample || non_canonical_reservoir.UCW <= 0.0f)
                continue;

            LightSampleInformation light_sample;
            light_sample.area_measure_pdf = 1.0f / non_canonical_reservoir.UCW;
            light_sample.emission = get_emission_of_triangle_from_index(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
            light_sample.emissive_triangle_index = non_canonical_reservoir.sample.emissive_triangle_index;
            light_sample.light_area = triangle_area(render_data, non_canonical_reservoir.sample.emissive_triangle_index);
            light_sample.light_source_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, non_canonical_reservoir.sample.emissive_triangle_index));
            light_sample.point_on_light = non_canonical_reservoir.sample.point_on_light;

            if (light_sample.area_measure_pdf <= 0.0f)
                // Can happen for very small triangles
                continue;

            float non_canonical_target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, hash_grid_cell_index, light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, random_number_generator);

            if (non_canonical_target_function <= 0.0f)
                continue;

            non_canonical_cell_integration_sum += non_canonical_target_function * non_canonical_reservoir.UCW;
        }

        regir_settings.non_canonical_pre_integration_factors[hash_grid_cell_index] += non_canonical_cell_integration_sum / (regir_settings.grid_fill.get_non_canonical_reservoir_count_per_cell() / 2) / REGIR_PRE_INTEGRATION_ITERATIONS;

        float canonical_cell_integration_sum = 0.0f;
        for (int i = 0; i < regir_settings.grid_fill.get_canonical_reservoir_count_per_cell() / 2; i++)
        {
            bool invalid_sample = false;
            ReGIRReservoir canonical_reservoir = regir_settings.get_random_cell_canonical_reservoir(representative_point, render_data.current_camera, roughness, random_number_generator, &invalid_sample);
            if (invalid_sample || canonical_reservoir.UCW <= 0.0f)
                continue;

            LightSampleInformation light_sample;
            light_sample.area_measure_pdf = 1.0f / canonical_reservoir.UCW;
            light_sample.emission = get_emission_of_triangle_from_index(render_data, canonical_reservoir.sample.emissive_triangle_index);
            light_sample.emissive_triangle_index = canonical_reservoir.sample.emissive_triangle_index;
            light_sample.light_area = triangle_area(render_data, canonical_reservoir.sample.emissive_triangle_index);
            light_sample.light_source_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, canonical_reservoir.sample.emissive_triangle_index));
            light_sample.point_on_light = canonical_reservoir.sample.point_on_light;

            if (light_sample.area_measure_pdf <= 0.0f)
                // Can happen for very small triangles
                continue;

            float canonical_target_function = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, hash_grid_cell_index, light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, random_number_generator);

            if (canonical_target_function <= 0.0f)
                continue;

            canonical_cell_integration_sum += canonical_target_function * canonical_reservoir.UCW;
        }

        regir_settings.canonical_pre_integration_factors[hash_grid_cell_index] += canonical_cell_integration_sum / (regir_settings.grid_fill.get_canonical_reservoir_count_per_cell() / 2) / REGIR_PRE_INTEGRATION_ITERATIONS;

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
