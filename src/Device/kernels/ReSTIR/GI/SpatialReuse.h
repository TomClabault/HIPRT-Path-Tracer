/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_ReSTIR_GI_SPATIAL_REUSE_H
#define DEVICE_ReSTIR_GI_SPATIAL_REUSE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/DI/Utils.h"
#include "Device/includes/ReSTIR/DI_GI/SpatialMISWeight.h"
#include "Device/includes/ReSTIR/DI_GI/SpatialNormalizationWeight.h"
#include "Device/includes/ReSTIR/DI_GI/UtilsSpatial.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/TargetFunction.h"
#include "Device/includes/ReSTIR/GI/Utils.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/** References:
 *
 * [1] [ReSTIR GI: Path Resampling for Real-Time Path Tracing] https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
 * [2] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
 * [3] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time - SIGGRAPH 2023 Presentation Video] https://dl.acm.org/doi/10.1145/3587423.3595511#sec-supp
 */

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_SpatialReuse(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_SpatialReuse(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t center_pixel_index = (x + y * render_data.render_settings.render_resolution.x);
	int2_t center_pixel_coords	= make_int2(x, y);

	if (!render_data.aux_buffers.pixel_active[center_pixel_index] || render_data.g_buffer.first_hit_prim_index[center_pixel_index] == -1)
	{
		// Pixel inactive because of adaptive sampling, returning
		// Or also we don't have a primary hit
		render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs[center_pixel_index] = ReSTIRGIReservoir();

		return;
	}

	// Initializing the random generator
	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(center_pixel_index));

	ReSTIRGIReservoir* input_reservoir_buffer = render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs;
	ReSTIRGIReservoir center_pixel_reservoir  = input_reservoir_buffer[center_pixel_index];
	// Surface data of the center pixel
	ReSTIRSurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index, random_number_generator);

	ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIR_VARIANT_GI>(render_data);
	// Generating a unique seed per pixel that will be used to generate (and replay for MIS weights) the spatial neighbors of that pixel
	spatial_pass_settings.spatial_neighbors_rng_seed = random_number_generator.xorshift32();

	setup_adaptive_directional_spatial_reuse<ReSTIR_VARIANT_GI>(render_data, center_pixel_index, random_number_generator);

	// Only used with MIS-like weight
	int selected_neighbor		  = 0;
	int neighbor_heuristics_cache = 0;
	int valid_neighbors_count	  = 0;
	int valid_neighbors_M_sum	  = 0;
	count_valid_spatial_neighbors<ReSTIR_VARIANT_GI>(render_data, center_pixel_surface, center_pixel_coords, valid_neighbors_count, valid_neighbors_M_sum,
													 neighbor_heuristics_cache);

	int reused_neighbors_count = render_data.render_settings.restir_gi_settings.common_spatial_pass.reuse_neighbor_count;
	int start_index			   = 0;
	if (valid_neighbors_M_sum == 0)
		// No valid neighbor to resample from, skip to the initial candidate right away
		start_index = reused_neighbors_count;

	ReSTIRGIReservoir spatial_reuse_output_reservoir;
	ReSTIRGISpatialResamplingMISWeight<ReSTIR_GI_MISWeightsType> mis_weight_function;
	Xorshift32Generator spatial_neighbors_rng(render_data.render_settings.restir_gi_settings.common_spatial_pass.spatial_neighbors_rng_seed);
	// Resampling the neighbors. Using neighbors + 1 here so that
	// we can use the last iteration of the loop to resample ourselves (the center pixel)
	//
	// See the implementation of get_spatial_neighbor_pixel_index() in ReSTIR/UtilsSpatial.h
	for (int neighbor_index = start_index; neighbor_index < reused_neighbors_count + 1; neighbor_index++)
	{
		const bool is_center_pixel = neighbor_index == reused_neighbors_count;

		int neighbor_pixel_index = get_spatial_neighbor_pixel_index<ReSTIR_VARIANT_GI>(render_data, neighbor_index, center_pixel_coords, spatial_neighbors_rng);
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		// We can already check whether or not this neighbor is going to be
		// accepted at all by checking the heuristic cache
		if (!is_center_pixel)
		{
			// If not the center pixel, we can check the heuristics, otherwise there's no need to,
			// we know that the center pixel will be accepted
			if ((neighbor_heuristics_cache & (1 << neighbor_index)) == 0)
				// Neighbor not passing the heuristics tests, skipping it right away
				continue;
		}

		ReSTIRGIReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];

		float shift_mapping_jacobian = 1.0f;
		if (neighbor_reservoir.UCW > 0.0f && !is_center_pixel && !neighbor_reservoir.sample.is_envmap_path())
		{
			// Only attempting the shift if the neighbor reservoir is valid
			//
			// Also, if this is the last neighbor resample (meaning that it is the center pixel),
			// the shift mapping is going to be an identity shift with a jacobian of 1 so we don't need to do it
			shift_mapping_jacobian = get_jacobian_determinant_reconnection_shift(
				neighbor_reservoir.sample.sample_point, neighbor_reservoir.sample.sample_point_geometric_normal.unpack(), center_pixel_surface.shading_point,
				render_data.g_buffer.primary_hit_position[neighbor_pixel_index],
				render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold());
		}

		float target_function_at_center				= 0.0f;
		bool do_neighbor_target_function_visibility = do_include_visibility_term_or_not<ReSTIR_VARIANT_GI>(render_data, neighbor_index);
		if (neighbor_reservoir.UCW > 0.0f)
		{
			if (is_center_pixel)
				// No need to evaluate the center sample at the center pixel, that's exactly
				// the target function of the center reservoir
				target_function_at_center = neighbor_reservoir.sample.target_function;
			else
			{
				if (do_neighbor_target_function_visibility)
					target_function_at_center = ReSTIR_GI_evaluate_target_function<KERNEL_OPTION_TRUE>(render_data, neighbor_reservoir.sample,
																									   center_pixel_surface, random_number_generator);
				else
					target_function_at_center = ReSTIR_GI_evaluate_target_function<KERNEL_OPTION_FALSE>(render_data, neighbor_reservoir.sample,
																										center_pixel_surface, random_number_generator);
			}
		}

#if ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(neighbor_reservoir.M);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(neighbor_reservoir.M);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir.M);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data,

																		 neighbor_reservoir.UCW, neighbor_reservoir.sample,

																		 center_pixel_surface, neighbor_index, center_pixel_coords, random_number_generator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS || ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE
		bool update_mc = center_pixel_reservoir.M > 0 && center_pixel_reservoir.UCW > 0.0f;

		float mis_weight = mis_weight_function.get_resampling_MIS_weight(
			render_data,

			neighbor_reservoir.M, neighbor_reservoir.sample.target_function, center_pixel_reservoir.sample, center_pixel_reservoir.M,
			center_pixel_reservoir.sample.target_function, neighbor_reservoir,

			center_pixel_surface, target_function_at_center * shift_mapping_jacobian, neighbor_pixel_index, valid_neighbors_count, valid_neighbors_M_sum,
			update_mc, /* resampling canonical */ is_center_pixel, random_number_generator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO || ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO
		bool update_mc = center_pixel_reservoir.M > 0 && center_pixel_reservoir.UCW > 0.0f;

		float mis_weight = mis_weight_function.get_resampling_MIS_weight(
			render_data,

			neighbor_reservoir.M, neighbor_reservoir.sample.target_function, center_pixel_reservoir.sample, center_pixel_reservoir.M,
			center_pixel_reservoir.sample.target_function, neighbor_reservoir,

			center_pixel_surface, target_function_at_center * shift_mapping_jacobian, neighbor_pixel_index, valid_neighbors_count, valid_neighbors_M_sum,
			update_mc, /* resampling canonical */ is_center_pixel, random_number_generator);
#else
#error "Unsupported mis weight type"
#endif

		// Combining as in Alg. 1 of the ReSTIR GI paper
		if (spatial_reuse_output_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, shift_mapping_jacobian,
														random_number_generator))
			// Only used with MIS-like MIS weights
			selected_neighbor = neighbor_index;

		spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);
	}

	float normalization_numerator	= 1.0f;
	float normalization_denominator = 1.0f;

	ReSTIRGISpatialNormalizationWeight<ReSTIR_GI_MISWeightsType> normalization_function;
#if ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M
	normalization_function.get_normalization(render_data, spatial_reuse_output_reservoir.weight_sum, center_pixel_surface, center_pixel_coords,
											 normalization_numerator, normalization_denominator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z
	normalization_function.get_normalization(render_data, spatial_reuse_output_reservoir.sample, spatial_reuse_output_reservoir.weight_sum,
											 center_pixel_surface, center_pixel_coords, normalization_numerator, normalization_denominator,
											 random_number_generator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE
	normalization_function.get_normalization(render_data, spatial_reuse_output_reservoir.sample, spatial_reuse_output_reservoir.weight_sum,
											 center_pixel_surface, selected_neighbor, center_pixel_coords, normalization_numerator, normalization_denominator,
											 random_number_generator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS || ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO || ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||                                                                           \
	ReSTIR_GI_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE
	// Stochastic pairwise MIS use their own custom spatial reuse kernel so this is just here so that everything compiles fine and we don't get the
	// "unsupported mis weight type" error
	float mis_weight = 1.0f;
#else
#error "Unsupported mis weight type"
#endif

	spatial_reuse_output_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);

	// Validating that the sample point resampled is visible from our visible point
	// TODO use a flag in the sample reservoir to indicate whether we are unoccluded or not
	//		(we are always unoccluded if we resampled the canonical sample for example, in which case we don't have to do the validation)
	//		It would also probably be beneficial to have another kernel do the validation such that samples that don't need the validation
	//		(resampled the canonical neighbor) don't do the validation at all
	ReSTIR_GI_visibility_validation(render_data, spatial_reuse_output_reservoir, center_pixel_surface.shading_point, center_pixel_surface.primitive_index,
									random_number_generator);

	// M-capping so that we don't have to M-cap when reading reservoirs on the next frame
	if (render_data.render_settings.restir_gi_settings.m_cap > 0)
		// M-capping the spatial neighbor if an M-cap has been given
		spatial_reuse_output_reservoir.M = hippt::min(spatial_reuse_output_reservoir.M, render_data.render_settings.restir_gi_settings.m_cap);

	render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs[center_pixel_index] = spatial_reuse_output_reservoir;
	render_data.store_updated_random_seed(center_pixel_index, random_number_generator.m_state.seed);
}

#endif
