/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_COMPUTE_CELLS_LIGHT_DISTRIBUTIONS_H
#define DEVICE_KERNELS_REGIR_COMPUTE_CELLS_LIGHT_DISTRIBUTIONS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/LightSampling/TriangleSampling.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/RenderData.h"

#if ReGIR_GridFillCellDistributionsIntegrateMesh == KERNEL_OPTION_TRUE

HIPRT_DEVICE float compute_mesh_contribution(HIPRTRenderData& render_data,
											 const ReGIRGridFillSurface& cell_surface,
											 unsigned int mesh_index_for_grid_cell,
											 bool primary_hit,
											 Xorshift32Generator& rng)
{
	EmissiveMeshAliasTableDevice mesh_alias_table = render_data.buffers.emissive_meshes_data.get_emissive_mesh_alias_table(mesh_index_for_grid_cell);
	float total_contribution_to_cell			  = 0.0f;
	for (int i = 0; i < ReGIR_GridFillCellDistributionsIntegrateMeshSampleCount; i++)
	{
		DeviceUnpackedEffectiveMaterial approximate_material;
		approximate_material.roughness = cell_surface.cell_roughness;
		approximate_material.metallic  = cell_surface.cell_metallic;
		approximate_material.specular  = cell_surface.cell_specular;

		float sample_PDF;
		int emissive_triangle_global_index			  = mesh_alias_table.sample_one_triangle_power(rng, sample_PDF);
		LightSamplePointInformation mesh_light_sample = sample_point_on_light_and_fill_light_sample_information(
								render_data, cell_surface.cell_point, hippt::normalize(render_data.current_camera.position - cell_surface.cell_point),
								cell_surface.cell_normal, approximate_material, emissive_triangle_global_index, rng);
		if (mesh_light_sample.emissive_triangle_global_index == -1)
			continue;

		sample_PDF *= mesh_light_sample.area_measure_pdf;

		total_contribution_to_cell += ReGIR_grid_fill_evaluate_target_function<
															  /* visibility */ false, ReGIR_GridFillTargetFunctionCosineTerm,
															  ReGIR_GridFillTargetFunctionCosineTermLightSource, ReGIR_GridFillPrimaryHitsTargetFunctionBSDF,
															  ReGIR_GridFillSecondaryHitsTargetFunctionBSDF,
															  ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation>(
															  render_data, cell_surface, primary_hit, mesh_light_sample.emission,
															  mesh_light_sample.light_source_normal, mesh_light_sample.point_on_light, rng) /
									  sample_PDF;
	}

	return total_contribution_to_cell / (float)ReGIR_GridFillCellDistributionsIntegrateMeshSampleCount;
}

#else // ReGIR_GridFillCellDistributionsIntegrateMesh == KERNEL_OPTION_TRUE

HIPRT_DEVICE float compute_mesh_contribution(HIPRTRenderData& render_data,
											 const ReGIRGridFillSurface& cell_surface,
											 unsigned int mesh_index_for_grid_cell,
											 bool primary_hit,
											 Xorshift32Generator& rng)
{
	float3_t mesh_average_point = render_data.buffers.emissive_meshes_data.meshes_average_points[mesh_index_for_grid_cell];
	float3_t mesh_normal;

#if ReGIR_GridFillCellDistributionsUseRepresentativeNormal == KERNEL_OPTION_TRUE
	float3_t mesh_average_normal = render_data.buffers.emissive_meshes_data.meshes_representative_normals[mesh_index_for_grid_cell];
	if (mesh_average_normal.x == EmissiveMeshesAliasTablesDevice::INVALID_NORMAL)
		// Invalid normal, using the direction from the cell to the mesh average point instead
		// such that the geometry term cosine term evaluates to 1 and the normal essentially
		// isn't taken into account
		mesh_normal = -hippt::normalize(mesh_average_point - cell_surface.cell_point);
	else
		mesh_normal = mesh_average_normal;
#else
	// If not using mesh representative normals, using the direction from the cell to the mesh average point instead
	// such that the geometry term cosine term evaluates to 1 and the normal essentially
	// isn't taken into account
	mesh_normal = -hippt::normalize(mesh_average_point - cell_surface.cell_point);
#endif

	// Just wrapping the mesh power in an RGB value to be able to pass it to the 'target_function' function which doesn't take
	// just a float as argument
	ColorRGB32F total_mesh_power = ColorRGB32F(render_data.buffers.emissive_meshes_data.meshes_total_power[mesh_index_for_grid_cell]);

	return ReGIR_grid_fill_evaluate_target_function<
							/* visibility */ false, ReGIR_GridFillTargetFunctionCosineTerm, ReGIR_GridFillTargetFunctionCosineTermLightSource,
							ReGIR_GridFillPrimaryHitsTargetFunctionBSDF, ReGIR_GridFillSecondaryHitsTargetFunctionBSDF,
							ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation>(render_data, cell_surface, primary_hit, total_mesh_power, mesh_normal,
																						 mesh_average_point, rng);
}

#endif // ReGIR_GridFillCellDistributionsIntegrateMesh

/**
 * This kernel computes the contribution of all the meshes of the scene to each grid cell
 * of the hash grid
 *
 * These contributions are then going to be used for building an alias table per each grid cell
 * which will be used to sample important emissive meshes directly, in one alias table sample
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReGIR_Compute_Cells_Light_Distributions(HIPRTRenderData render_data,
										unsigned int* contributions_scratch_buffer_sort_keys,
										unsigned int* contributions_scratch_buffer_sort_values,
										unsigned int cell_index_offset,
										bool primary_hit)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReGIR_Compute_Cells_Light_Distributions(HIPRTRenderData render_data,
											   unsigned int* contributions_scratch_buffer_sort_keys,
											   unsigned int* contributions_scratch_buffer_sort_values,
											   unsigned int cell_index_offset,
											   bool primary_hit,
											   unsigned int thread_index)
#endif
{
	if (render_data.buffers.emissive_triangles_count == 0)
		// No initial candidates to sample since no lights
		return;

	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
	uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int emissive_mesh_count = render_data.buffers.emissive_meshes_data.alias_table_count;
	// Cell index within the dispatch
	unsigned int local_cell_index = thread_index / emissive_mesh_count;
	// Global cell index in [0, total number of grid cells in the scene]
	unsigned int global_cell_index = cell_index_offset + local_cell_index;
	if (global_cell_index >= hippt::atomic_load(regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_count))
		// Compute shader threads are outside the valid range
		return;
	unsigned int hash_grid_cell_index	  = regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_list[global_cell_index];
	unsigned int mesh_index_for_grid_cell = thread_index % emissive_mesh_count;

	ReGIRGridFillSurface cell_surface = ReGIR_get_cell_surface(render_data, hash_grid_cell_index, primary_hit);

	unsigned int seed = wang_hash(hash_grid_cell_index + mesh_index_for_grid_cell);
	Xorshift32Generator rng(seed);

	float mesh_contribution = compute_mesh_contribution(render_data, cell_surface, mesh_index_for_grid_cell, primary_hit, rng);
	// Using log luminance to compress very high values and help precision of fp16
	float log_mesh_contribution		= hippt::intrin_logf(mesh_contribution + 1.0f);
	fp16 log_mesh_contribution_fp16 = static_cast<fp16>(log_mesh_contribution);

	// We're producing a sort key here so that we can sort mesh indices by contribution after this kernel.
	// The sort key is made of the contribution in the lower 16 bits and the local cell index in the upper 16 bits. We need the local cell index to be able to
	// regroup the contributions by cell after sorting (since contributions of different cells are mixed together in the scratch buffer) and we need the
	// contribution in the lower bits because we want to sort by contribution first.
	contributions_scratch_buffer_sort_keys[thread_index] =
							static_cast<unsigned int>((~hippt::half_as_ushort(log_mesh_contribution_fp16)) & 0xFFFF) | (local_cell_index << 16);
	// This is just going to be mesh indices in [0, emissive_mesh_count - 1], per each grid cell
	contributions_scratch_buffer_sort_values[thread_index] = mesh_index_for_grid_cell;
}

#endif
