/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_GRID_FILL_H
#define DEVICE_KERNELS_REGIR_GRID_FILL_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/TriangleEmissiveSampling.h"
#include "Device/includes/ReSTIR/ReGIR/LightDistributionsGridFill.h"
#include "Device/includes/ReSTIR/ReGIR/LightDistributionsGridFillSampleCanonical.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/TargetFunction.h"

#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyCanonical>()> grid_fill_sample_canonical_candidate(
						const HIPRTRenderData& render_data, const ReGIRGridFillSurface& surface, float3_t view_direction, Xorshift32Generator& rng)
{
	RayPayload dummy_ray_payload;
	dummy_ray_payload.material.roughness = surface.cell_roughness;
	dummy_ray_payload.material.metallic	 = surface.cell_metallic;
	dummy_ray_payload.material.specular	 = surface.cell_specular;

#if ReGIR_GridFillLightSamplingBaseStrategyCanonical == LSS_BASE_LIGHT_TREE_ATS
	return sample_one_emissive_triangle_light_tree_ats<false>(render_data, surface.cell_point, view_direction, surface.cell_normal, surface.cell_normal,
															  surface.cell_primitive_index, dummy_ray_payload, rng);
#else
	return sample_one_point_on_light<ReGIR_GridFillLightSamplingBaseStrategyCanonical>(render_data, surface.cell_point, view_direction, surface.cell_normal,
																					   surface.cell_normal, surface.cell_primitive_index, dummy_ray_payload,
																					   rng);
#endif
}

HIPRT_DEVICE ReGIRReservoir grid_fill_with_per_cell_light_distributions(const HIPRTRenderData& render_data,
																		unsigned int hash_grid_cell_index,
																		int reservoir_index_in_cell,
																		const ReGIRGridFillSurface& surface,
																		bool primary_hit,
																		Xorshift32Generator& rng)
{
	ReGIRReservoir reservoir;

	const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
	bool reservoir_is_canonical			= regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell);

	// Sampling some samples with per-cell light distributions
	for (int light_sample_index = 0; light_sample_index < regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir;
		 light_sample_index++)
	{
		if (reservoir_is_canonical)
		{
			LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyCanonical>()> light_point_samples =
									grid_fill_sample_canonical_candidate(render_data, surface,
																		 hippt::normalize(render_data.current_camera.position - surface.cell_point), rng);

			for (int i = 0; i < DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyCanonical>(); i++)
			{
				LightSamplePointInformation& light_point_sample = light_point_samples[i];
				if (light_point_sample.emissive_triangle_global_index == -1)
					continue;

				// This reservoir is canonical, simple target function to keep it canonical (no visibility / cosine terms)
				float target_function = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, surface, primary_hit, light_point_sample.emission,
																						   light_point_sample.light_source_normal,
																						   light_point_sample.point_on_light, rng);

				float mis_weight = 1.0f / (regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir *
										   DirectLightIntegrationFactor<ReGIR_GridFillLightSamplingBaseStrategyCanonical>());

				reservoir.stream_sample(mis_weight, target_function, light_point_sample.area_measure_pdf, light_point_sample, rng);
				sanity_check<true>(render_data, reservoir.weight_sum, -1, -1);
			}
		}
		else
		{
			DeviceUnpackedEffectiveMaterial material;
			material.roughness = surface.cell_roughness;
			material.metallic  = surface.cell_metallic;
			material.specular  = surface.cell_specular;

			LightSamplePointInformation light_sample = sample_one_emissive_triangle_with_cell_light_distribution(
									render_data, surface.cell_point, hippt::normalize(render_data.current_camera.position - surface.cell_point),
									surface.cell_normal, material, hash_grid_cell_index, primary_hit, rng);

			// TODO DO WE NEED THIS
			/*if (light_sample.emissive_triangle_global_index == REGIR_NEEDS_LIGHT_SAMPLE_FALLBACK)
				light_sample = grid_fill_sample_canonical_candidate(render_data, surface, hippt::normalize(render_data.current_camera.position -
			   surface.cell_point), rng);*/

			if (light_sample.emissive_triangle_global_index == -1)
				continue;

			float target_function = ReGIR_grid_fill_evaluate_target_function < ReGIR_GridFillTargetFunctionVisibility, ReGIR_GridFillTargetFunctionCosineTerm,
				  ReGIR_GridFillTargetFunctionCosineTermLightSource, ReGIR_GridFillPrimaryHitsTargetFunctionBSDF, ReGIR_GridFillSecondaryHitsTargetFunctionBSDF,
				  /* We don't need NEE++ here because it's already included in the sampling distribution of the grid cell.
					 We don't need that in RIS*/
									ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation &&
															ReGIR_GridFillCellDistributionsUnbiasedNEEPlusPlus >
																					(render_data, surface, primary_hit, light_sample.emission,
																					 light_sample.light_source_normal, light_sample.point_on_light, rng);

			float simple_strategy_PDF = pdf_of_emissive_triangle_hit_area_measure<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(
									render_data, surface.cell_point, hippt::normalize(render_data.current_camera.position - surface.cell_point),
									surface.cell_normal, material, light_sample.point_on_light, light_sample.light_source_normal,
									light_sample.emissive_triangle_global_index, light_sample.light_area, light_sample.emission);
			float mis_weight = balance_heuristic(
									light_sample.area_measure_pdf, regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir,
									simple_strategy_PDF,
									ReGIR_GridFillCellDistributionsCanonicalSampleCount *
															DirectLightIntegrationFactor<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>());

			reservoir.stream_sample(mis_weight, target_function, light_sample.area_measure_pdf, light_sample, rng);
			sanity_check<true>(render_data, reservoir.weight_sum, -1, -1);
		}
	}

	if (!reservoir_is_canonical)
	{
		// Sampling some samples with a simple 'cover-all-triangles" strategy (power sampling for example)
		// for unbiasedness because the per-cell light distributions are expected by the shading
		// kernel to be able to cover all lights in the scene
		for (int light_sample_index = 0; light_sample_index < ReGIR_GridFillCellDistributionsCanonicalSampleCount; light_sample_index++)
		{
			LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>()> light_point_samples =
									grid_fill_cell_light_distributions_canonical_sample(
															render_data, surface, hippt::normalize(render_data.current_camera.position - surface.cell_point),
															rng);

			for (int i = 0; i < DirectLightSampleCount<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(); i++)
			{
				LightSamplePointInformation& light_point_sample = light_point_samples[i];
				if (light_point_sample.emissive_triangle_global_index == -1)
					// Can happen if the triangle sampled is degenerate (for example) and thus rejected
					// during sampling
					continue;

				float target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, surface, primary_hit, light_point_sample.emission,
																							   light_point_sample.light_source_normal,
																							   light_point_sample.point_on_light, rng);
				unsigned int sampled_mesh_index = render_data.buffers.emissive_meshes_data.global_triangle_index_to_emissive_mesh_index
																		  [light_point_sample.emissive_triangle_global_index];
				float cell_light_distributions_pdf = get_cell_distribution_PDF_of_light_sample(render_data, hash_grid_cell_index, primary_hit,
																							   light_point_sample, sampled_mesh_index);
				float mis_weight				   = balance_heuristic(
										  light_point_sample.area_measure_pdf,
										  ReGIR_GridFillCellDistributionsCanonicalSampleCount *
																  DirectLightIntegrationFactor<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(),
										  cell_light_distributions_pdf, regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir);

				reservoir.stream_sample(mis_weight, target_function, light_point_sample.area_measure_pdf, light_point_sample, rng);
				sanity_check<true>(render_data, reservoir.weight_sum, -1, -1);
			}
		}
	}

	return reservoir;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE ReGIRReservoir grid_fill_classic(const HIPRTRenderData& render_data,
											  unsigned int hash_grid_cell_index,
											  int reservoir_index_in_cell,
											  const ReGIRGridFillSurface& surface,
											  bool primary_hit,
											  Xorshift32Generator& rng)
{
	ReGIRReservoir reservoir;

	const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
	bool reservoir_is_canonical			= regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell);

	for (int light_sample_index = 0; light_sample_index < regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir;
		 light_sample_index++)
	{
		if (reservoir_is_canonical)
		{
			LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyCanonical>()> light_point_samples =
									grid_fill_sample_canonical_candidate(render_data, surface,
																		 hippt::normalize(render_data.current_camera.position - surface.cell_point), rng);

			for (int i = 0; i < DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyCanonical>(); i++)
			{
				LightSamplePointInformation& light_point_sample = light_point_samples[i];
				if (light_point_sample.emissive_triangle_global_index == -1)
					continue;

				float target_function = ReGIR_grid_fill_evaluate_canonical_target_function(render_data, surface, primary_hit, light_point_sample.emission,
																						   light_point_sample.light_source_normal,
																						   light_point_sample.point_on_light, rng);

				float mis_weight = 1.0f / (regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir *
										   DirectLightIntegrationFactor<ReGIR_GridFillLightSamplingBaseStrategyCanonical>());
				float source_pdf = light_point_sample.area_measure_pdf;

				sanity_check<true>(render_data, source_pdf, -1, -1);
				sanity_check<true>(render_data, target_function, -1, -1);
				reservoir.stream_sample(mis_weight, target_function, source_pdf, light_point_sample, rng);
			}
		}
		else
		{
			RayPayload dummy_ray_payload;
			dummy_ray_payload.material.roughness = surface.cell_roughness;
			dummy_ray_payload.material.metallic	 = surface.cell_metallic;
			dummy_ray_payload.material.specular	 = surface.cell_specular;

			LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyNonCanonical>()> light_point_samples =
									sample_one_point_on_light<ReGIR_GridFillLightSamplingBaseStrategyNonCanonical>(
															render_data, surface.cell_point,
															hippt::normalize(render_data.current_camera.position - surface.cell_point), surface.cell_normal,
															surface.cell_normal, surface.cell_primitive_index, dummy_ray_payload, rng);

			for (int i = 0; i < DirectLightSampleCount<ReGIR_GridFillLightSamplingBaseStrategyNonCanonical>(); i++)
			{
				LightSamplePointInformation& light_point_sample = light_point_samples[i];
				if (light_point_sample.emissive_triangle_global_index == -1)
					continue;

				float target_function = ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, surface, primary_hit, light_point_sample.emission,
																							   light_point_sample.light_source_normal,
																							   light_point_sample.point_on_light, rng);
				float mis_weight	  = 1.0f / (regir_settings.get_grid_fill_settings(primary_hit).light_sample_count_per_cell_reservoir *
											DirectLightIntegrationFactor<ReGIR_GridFillLightSamplingBaseStrategyNonCanonical>());
				float source_pdf	  = light_point_sample.area_measure_pdf;

				sanity_check<true>(render_data, source_pdf, -1, -1);
				sanity_check<true>(render_data, target_function, -1, -1);
				reservoir.stream_sample(mis_weight, target_function, source_pdf, light_point_sample, rng);
			}
		}
	}

	return reservoir;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE ReGIRReservoir grid_fill(const HIPRTRenderData& render_data,
									  const ReGIRSettings& regir_settings,
									  unsigned int hash_grid_cell_index,
									  int reservoir_index_in_cell,
									  const ReGIRGridFillSurface& surface,
									  bool primary_hit,
									  Xorshift32Generator& rng)
{
	ReGIRReservoir grid_fill_reservoir;

	if constexpr (ReGIR_GridFillUsePerCellLightDistributions == KERNEL_OPTION_TRUE)
		grid_fill_reservoir = grid_fill_with_per_cell_light_distributions(render_data, hash_grid_cell_index, reservoir_index_in_cell, surface, primary_hit,
																		  rng);
	else
		grid_fill_reservoir = grid_fill_classic<accumulatePreIntegration>(render_data, hash_grid_cell_index, reservoir_index_in_cell, surface, primary_hit,
																		  rng);

	return grid_fill_reservoir;
}

template <bool accumulatePreIntegration>
HIPRT_DEVICE void grid_fill_pre_integration_accumulation(HIPRTRenderData& render_data,
														 const ReGIRReservoir& output_reservoir,
														 bool reservoir_is_canonical,
														 unsigned int hash_grid_cell_index,
														 bool primary_hit)
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
				normalization = regir_settings.get_grid_fill_settings(primary_hit).get_canonical_reservoir_count_per_cell() *
								render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS;
			else
				normalization = regir_settings.get_grid_fill_settings(primary_hit).get_non_canonical_reservoir_count_per_cell() *
								render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS;
			float integration_increment = hippt::max(0.0f, output_reservoir.sample.target_function * output_reservoir.UCW) / normalization;

			if (reservoir_is_canonical)
				hippt::atomic_fetch_add(&regir_settings.get_canonical_pre_integration_factor_buffer(primary_hit)[hash_grid_cell_index], integration_increment);
			else
				hippt::atomic_fetch_add(&regir_settings.get_non_canonical_pre_integration_factor_buffer(primary_hit)[hash_grid_cell_index],
										integration_increment);
		}
	}
}

/**
 * This kernel is in charge of resetting (when necessary) and filling the ReGIR grid.
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(64) ReGIR_Grid_Fill(HIPRTRenderData render_data,
									  ReGIRHashGridSoADevice output_reservoirs_grid,
									  unsigned int number_of_cells_alive,
									  bool primary_hit)
#else
template <bool accumulatePreIntegration>
GLOBAL_KERNEL_SIGNATURE(void)
inline ReGIR_Grid_Fill(HIPRTRenderData render_data,
					   ReGIRHashGridSoADevice output_reservoirs_grid,
					   unsigned int number_of_cells_alive,
					   bool primary_hit,
					   int thread_index)
#endif
{
	if (render_data.buffers.emissive_triangles_count == 0)
		// No initial candidates to sample since no lights
		return;

	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#ifdef __KERNELCC__
	uint32_t thread_index		= blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t thread_count = gridDim.x * blockDim.x;
#endif

	while (thread_index < regir_settings.get_number_of_reservoirs_per_cell(primary_hit) * number_of_cells_alive)
	{
		int reservoir_index = thread_index;

		unsigned int reservoir_index_in_cell = reservoir_index % regir_settings.get_number_of_reservoirs_per_cell(primary_hit);
		unsigned int cell_alive_index		 = reservoir_index / regir_settings.get_number_of_reservoirs_per_cell(primary_hit);
		// If all cells are alive, the cell index is straightforward
		//
		// Not all cells are alive, what we have is cell_alive_index which is the index of the cell in the alive list
		// so we can fetch the index of the cell in the grid cells alive list with that cell_alive_index
		unsigned int hash_grid_cell_index	 = regir_settings.get_hash_cell_data_soa(primary_hit).grid_cells_alive_list[cell_alive_index];
		unsigned int reservoir_index_in_grid = hash_grid_cell_index * regir_settings.get_number_of_reservoirs_per_cell(primary_hit) + reservoir_index_in_cell;

		Xorshift32Generator random_number_generator((reservoir_index_in_grid + 1) * (render_data.render_settings.sample_number + 1) *
													render_data.render_settings.random_number);
		ReGIRReservoir output_reservoir;

		ReGIRGridFillSurface cell_surface = ReGIR_get_cell_surface(render_data, hash_grid_cell_index, primary_hit);

		// Grid fill
#ifdef __KERNELCC__
		constexpr bool ACCUMULATE_PRE_INTEGRATION_OPTION = ReGIR_GridFillSpatialReuse_AccumulatePreIntegration;
#else
		constexpr bool ACCUMULATE_PRE_INTEGRATION_OPTION = accumulatePreIntegration;
#endif
		output_reservoir = grid_fill<ACCUMULATE_PRE_INTEGRATION_OPTION>(render_data, regir_settings, hash_grid_cell_index, reservoir_index_in_cell,
																		cell_surface, primary_hit, random_number_generator);

		// Normalizing the reservoir
		output_reservoir.finalize_resampling(1.0f, 1.0f);
		sanity_check<true>(render_data, output_reservoir.UCW, -1, -1);

		regir_settings.store_reservoir_custom_buffer_opt(output_reservoirs_grid, output_reservoir, hash_grid_cell_index, reservoir_index_in_cell);

		grid_fill_pre_integration_accumulation<ACCUMULATE_PRE_INTEGRATION_OPTION>(
								render_data, output_reservoir,
								regir_settings.get_grid_fill_settings(primary_hit).reservoir_index_in_cell_is_canonical(reservoir_index_in_cell),
								hash_grid_cell_index, primary_hit);

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
