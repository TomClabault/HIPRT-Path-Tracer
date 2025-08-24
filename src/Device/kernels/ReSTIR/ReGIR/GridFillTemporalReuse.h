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

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_per_cell_distributions(const HIPRTRenderData& render_data, float3 surface_point, unsigned int hash_grid_cell_index, bool primary_hit, Xorshift32Generator& rng)
{
    const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    AliasTableDevice cell_alias_table = regir_settings.get_cell_alias_table(hash_grid_cell_index, primary_hit);
    int alias_table_index = cell_alias_table.sample(rng);
    unsigned int alias_table_size = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).alias_table_size;

    unsigned int emissive_mesh_index = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).emissive_meshes_indices[hash_grid_cell_index * alias_table_size + alias_table_index];
    float mesh_PDF = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).all_alias_tables_PDFs[hash_grid_cell_index * alias_table_size + alias_table_index];
    if (mesh_PDF == 0.0f)
        // No valid mesh for this cell, early exit by returning
        // an empty sample
        return LightSampleInformation();

    EmissiveMeshAliasTableDevice mesh_alias_table = render_data.buffers.emissive_meshes_data.get_emissive_mesh_alias_table(emissive_mesh_index);

    //// Now that we have importance sampled a mesh, we're importance sampling a triangle
    //// on that mesh
    //float triangle_PDF;
    //int emissive_triangle_index = mesh_alias_table.sample_one_triangle_power(rng, triangle_PDF);
    /*if (hash_grid_cell_index == 34720)
        std::cout << std::endl;*/

    // Total weight for choosing a bin (bin of faces binned by normal direction) to sample with WRS
	float3 average_mesh_point = render_data.buffers.emissive_meshes_data.meshes_average_points[emissive_mesh_index];
    float3 direction_to_mesh = hippt::normalize(average_mesh_point - surface_point);
    float total_weight = 0.0f;
	int selected_bin = -1;
    float selected_bin_weight = 0.0f;

    float triangle_PDF = 0.0f;
    // Doing WRS on the binned faces of the mesh to select a bin proportional to its approximate contribution
    // to the cell
    for (int i = 0; i < EmissiveMeshesDataDevice::BIN_NORMAL_COUNT; i++)
    {
		float bin_power = render_data.buffers.emissive_meshes_data.binned_faces_total_power[emissive_mesh_index * EmissiveMeshesDataDevice::BIN_NORMAL_COUNT + i];
        float bin_weight = bin_power * hippt::max(0.0f, hippt::dot(direction_to_mesh, -EmissiveMeshesDataDevice::get_binning_normal(i)));

        if (49277 == hash_grid_cell_index && emissive_mesh_index == 0 && i == 0)
            printf("binPower: %f\n", bin_power);

        if (rng() < bin_weight / total_weight)
        {
            selected_bin = i;
            selected_bin_weight = bin_weight;
        }

        total_weight += bin_weight;
    }

    if (total_weight == 0.0f)
        // The mesh is totally backfacing
        return LightSampleInformation();

    unsigned int bin_face_count = render_data.buffers.emissive_meshes_data.binned_faces_counts[emissive_mesh_index * EmissiveMeshesDataDevice::BIN_NORMAL_COUNT + selected_bin];
	triangle_PDF = selected_bin_weight / total_weight;
	triangle_PDF *= 1.0f / bin_face_count;
    unsigned int emissive_mesh_offset = render_data.buffers.emissive_meshes_data.offsets[emissive_mesh_index];
	unsigned int triangle_index_in_mesh = render_data.buffers.emissive_meshes_data.binned_faces_indices[emissive_mesh_offset + render_data.buffers.emissive_meshes_data.binned_faces_start_index[emissive_mesh_index * EmissiveMeshesDataDevice::BIN_NORMAL_COUNT + selected_bin] + rng.random_index(bin_face_count)];
    int emissive_triangle_index = render_data.buffers.emissive_meshes_data.meshes_triangle_indices[emissive_mesh_offset + triangle_index_in_mesh];

    /*if (49277 == hash_grid_cell_index)
        printf("meshIdx [%u], slect / total / bin face count: %f / %f / %u\n", emissive_mesh_index, selected_bin_weight, total_weight, bin_face_count);*/

    LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
    // Area measure PDF already contains the PDF for sampling the point *on the triangle*.
    // We need to add (multiply) the PDF of sampling the triangle itself within the sampled mesh
    light_sample.area_measure_pdf *= mesh_PDF * triangle_PDF;

    sanity_check<true>(render_data, ColorRGB32F(1.0f / light_sample.area_measure_pdf), -1, -1);

    return light_sample;
}

HIPRT_DEVICE float get_cell_distribution_PDF_of_light_sample(const HIPRTRenderData& render_data, unsigned int hash_grid_cell_index, bool primary_hit, const LightSampleInformation& light_sample, unsigned int emissive_mesh_index, unsigned int face_index_in_mesh, float3 surface_point,
    Xorshift32Generator& rng)
{
    const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    AliasTableDevice cell_alias_table = regir_settings.get_cell_alias_table(hash_grid_cell_index, primary_hit);
    unsigned int alias_table_size = render_data.render_settings.regir_settings.get_cell_distributions_soa(primary_hit).alias_table_size;

    // PDF that the light distribution of the cell samples that mesh
    float mesh_sampling_PDF = 0.0f;
    for (int i = 0; i < cell_alias_table.size; i++)
    {
        if (regir_settings.get_cell_distributions_soa(primary_hit).emissive_meshes_indices[hash_grid_cell_index * alias_table_size + i] == emissive_mesh_index)
        {
            mesh_sampling_PDF = regir_settings.get_cell_distributions_soa(primary_hit).all_alias_tables_PDFs[hash_grid_cell_index * alias_table_size + i];

            break;
        }
    }

    if (mesh_sampling_PDF == 0.0f)
        return 0.0f;

    float triangle_within_mesh_sampling_PDF = render_data.buffers.emissive_meshes_data.get_binned_faces_sampled_triangle_PDF_in_mesh(emissive_mesh_index, face_index_in_mesh, render_data.buffers.emissive_meshes_data.meshes_average_points[emissive_mesh_index], surface_point);
    float point_on_triangle_PDF = 1.0f / light_sample.light_area;

    return mesh_sampling_PDF * triangle_within_mesh_sampling_PDF * point_on_triangle_PDF;
}

HIPRT_DEVICE LightSampleInformation sample_one_presampled_light(const HIPRTRenderData& render_data, 
    unsigned int hash_grid_cell_index, int reservoir_index_in_cell, bool primary_hit,
    Xorshift32Generator& rng)
{
    const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

    float presampled_light_pdf;
    ReGIRPresampledLight light_sample = regir_settings.sample_one_presampled_light(hash_grid_cell_index, reservoir_index_in_cell, primary_hit, presampled_light_pdf, rng);

    LightSampleInformation full_sample_information;
    full_sample_information.emissive_triangle_index = light_sample.emissive_triangle_index;
    full_sample_information.light_source_normal = light_sample.normal.unpack();
     full_sample_information.light_area = light_sample.triangle_area;
    //full_sample_information.emission = light_sample.emission;
    full_sample_information.emission = render_data.buffers.materials_buffer.get_emission(render_data.buffers.material_indices[light_sample.emissive_triangle_index]);
    full_sample_information.point_on_light = light_sample.point_on_light;

    // PDF of that point on that triangle
    full_sample_information.area_measure_pdf = 1.0f / full_sample_information.light_area;
#if ReGIR_GridFillLightSamplingBaseStrategy == LSS_BASE_UNIFORM
    // PDF of sampling that triangle uniformly
    full_sample_information.area_measure_pdf *= 1.0f / render_data.buffers.emissive_triangles_count;
#elif ReGIR_GridFillLightSamplingBaseStrategy == LSS_BASE_POWER
    // PDF of sampling that triangle according to its power
    full_sample_information.area_measure_pdf *= (full_sample_information.emission.luminance() * full_sample_information.light_area) / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;
#endif

    return full_sample_information;
}

HIPRT_DEVICE ReGIRReservoir grid_fill_with_per_cell_light_distributions(const HIPRTRenderData& render_data,
    unsigned int hash_grid_cell_index, int reservoir_index_in_cell, const ReGIRGridFillSurface& surface, bool primary_hit,
    Xorshift32Generator& rng)
{
    ReGIRReservoir reservoir;
    
    const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
    bool reservoir_is_canonical = regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell);

    // Sampling some samples with per-cell light distributions
    for (int light_sample_index = 0; light_sample_index < regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir; light_sample_index++)
    {
        LightSampleInformation light_sample;

        if (reservoir_is_canonical)
            light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);
        else
            light_sample = sample_one_emissive_triangle_per_cell_distributions(render_data, surface.cell_point, hash_grid_cell_index, primary_hit, rng);

        if (light_sample.emissive_triangle_index == -1)
            continue;

        float target_function;
        if (reservoir_is_canonical)
            // This reservoir is canonical, simple target function to keep it canonical (no visibility / cosine terms)
            target_function = ReGIR_grid_fill_evaluate_canonical_target_function(render_data,
                surface, primary_hit,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        else
            target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data,
                surface, primary_hit,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);

        float mis_weight;
        if (reservoir_is_canonical)
            mis_weight = 1.0f / regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir;
        else
        {
            float simple_strategy_PDF = pdf_of_emissive_triangle_hit_area_measure<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, light_sample.light_area, light_sample.emission);
            mis_weight = balance_heuristic(light_sample.area_measure_pdf, regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir, simple_strategy_PDF, ReGIR_GridFillPerCellDistributionsCanonicalSampleCount);
        }

        reservoir.stream_sample(mis_weight, target_function, light_sample.area_measure_pdf, light_sample, rng);
    }

    if (!reservoir_is_canonical)
    {
        // Sampling some samples with a simple 'cover-all-triangles" strategy (power sampling for example)
        // for unbiasedness
        for (int light_sample_index = 0; light_sample_index < ReGIR_GridFillPerCellDistributionsCanonicalSampleCount; light_sample_index++)
        {
            float mesh_PDF;
            unsigned int mesh_index;
            EmissiveMeshAliasTableDevice mesh_alias_table = render_data.buffers.emissive_meshes_data.sample_one_emissive_mesh(rng, mesh_PDF, mesh_index);

            float triangle_PDF;
            unsigned triangle_index_in_mesh;
            int emissive_triangle_index = mesh_alias_table.sample_one_triangle_power(rng, triangle_index_in_mesh, triangle_PDF);
            if (emissive_triangle_index == -1)
                continue;

            LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
            // That point on this triangle on that emissive mesh
            light_sample.area_measure_pdf *= mesh_PDF * triangle_PDF;

            float target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data,
                surface, primary_hit,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
            float cell_light_distributions_pdf = get_cell_distribution_PDF_of_light_sample(render_data, hash_grid_cell_index, primary_hit, light_sample, mesh_index, triangle_index_in_mesh, surface.cell_point, rng);
            float mis_weight = balance_heuristic(light_sample.area_measure_pdf, ReGIR_GridFillPerCellDistributionsCanonicalSampleCount, cell_light_distributions_pdf, regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir);

            reservoir.stream_sample(mis_weight, target_function, light_sample.area_measure_pdf, light_sample, rng);
        }
    }

    return reservoir;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE ReGIRReservoir grid_fill_classic(const HIPRTRenderData& render_data,
    unsigned int hash_grid_cell_index, int reservoir_index_in_cell, const ReGIRGridFillSurface& surface, bool primary_hit,
    Xorshift32Generator& rng)
{
    ReGIRReservoir reservoir;

    const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
    bool reservoir_is_canonical = regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell);

    for (int light_sample_index = 0; light_sample_index < regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir; light_sample_index++)
    {
        LightSampleInformation light_sample;

        if constexpr (ReGIR_GridFillDoLightPresampling == KERNEL_OPTION_TRUE && !accumulatePreIntegration)
            // Never using presampling lights for pre integration because pre integration needs
            // different samples to pre integrate properly and using presampled lights severely restricts
            // the number of different samples we have available
            light_sample = sample_one_presampled_light(render_data, hash_grid_cell_index, reservoir_index_in_cell, primary_hit, rng);
        else
            light_sample = sample_one_emissive_triangle<ReGIR_GridFillLightSamplingBaseStrategy>(render_data, rng);

        if (light_sample.emissive_triangle_index == -1)
            continue;

        float target_function;
        if (reservoir_is_canonical)
            // This reservoir is canonical, simple target function to keep it canonical (no visibility / cosine terms)
            target_function = ReGIR_grid_fill_evaluate_canonical_target_function(render_data,
                surface, primary_hit,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);
        else
            target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data,
                surface, primary_hit,
                light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, rng);

        float mis_weight = 1.0f / regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir;
        float source_pdf = light_sample.area_measure_pdf;

        reservoir.stream_sample(mis_weight, target_function, source_pdf, light_sample, rng);
    }

    return reservoir;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE ReGIRReservoir grid_fill(const HIPRTRenderData& render_data, const ReGIRSettings& regir_settings,
    unsigned int hash_grid_cell_index, int reservoir_index_in_cell, const ReGIRGridFillSurface& surface, bool primary_hit,
    Xorshift32Generator& rng)
{
    ReGIRReservoir grid_fill_reservoir;

    if constexpr (ReGIR_GridFillUsePerCellDistributions == KERNEL_OPTION_TRUE)
        grid_fill_reservoir = grid_fill_with_per_cell_light_distributions(render_data, hash_grid_cell_index, reservoir_index_in_cell, surface, primary_hit, rng);
    else
        grid_fill_reservoir = grid_fill_classic<accumulatePreIntegration>(render_data, hash_grid_cell_index, reservoir_index_in_cell, surface, primary_hit, rng);

    sanity_check<true>(render_data, grid_fill_reservoir.weight_sum);
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
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, ReGIRHashGridSoADevice output_reservoirs_grid, unsigned int number_of_cells_alive, bool primary_hit)
#else
template <bool accumulatePreIntegration>
GLOBAL_KERNEL_SIGNATURE(void) inline ReGIR_Grid_Fill_Temporal_Reuse(HIPRTRenderData render_data, ReGIRHashGridSoADevice output_reservoirs_grid, unsigned int number_of_cells_alive, bool primary_hit, int thread_index)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0)
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
        
        unsigned int seed;
        if (render_data.render_settings.freeze_random)
            seed = wang_hash(reservoir_index_in_grid + 1);
        else
            seed = wang_hash((reservoir_index_in_grid + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
        
        Xorshift32Generator random_number_generator(seed);
        ReGIRReservoir output_reservoir;

        ReGIRGridFillSurface cell_surface = ReGIR_get_cell_surface(render_data, hash_grid_cell_index, primary_hit);

        // Grid fill
#ifdef __KERNELCC__
        constexpr bool ACCUMULATE_PRE_INTEGRATION_OPTION = ReGIR_GridFillSpatialReuse_AccumulatePreIntegration;
#else
        constexpr bool ACCUMULATE_PRE_INTEGRATION_OPTION = accumulatePreIntegration;
#endif
        output_reservoir = grid_fill<ACCUMULATE_PRE_INTEGRATION_OPTION>(render_data, regir_settings, hash_grid_cell_index, reservoir_index_in_cell, cell_surface, primary_hit, random_number_generator);
        
        // Normalizing the reservoir
        output_reservoir.finalize_resampling(1.0f, 1.0f);
        
        regir_settings.store_reservoir_custom_buffer_opt(output_reservoirs_grid, output_reservoir, hash_grid_cell_index, reservoir_index_in_cell);

        grid_fill_pre_integration_accumulation<ACCUMULATE_PRE_INTEGRATION_OPTION>(render_data, output_reservoir, regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell), hash_grid_cell_index, primary_hit);

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
