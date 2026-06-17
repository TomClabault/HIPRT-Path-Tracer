/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/DI_GI/UtilsSpatial.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/SpatialMISWeight.h"
#include "Device/includes/ReSTIR/PT/SpatialNormalizationWeight.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/PT/Utils.h"
#include "Device/includes/ReSTIR/PT/UtilsSpatialSPMIS.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

#define DO_DEBUG_CONDITION 0

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_SpatialReuseSPMIS(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SpatialReuseSPMIS(HIPRTRenderData render_data, int x, int y)
#endif
{
	// Only compiling this whole kernel if we're using stochastic pairwise MIS
#if ReSTIR_PT_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||                                                                             \
	ReSTIR_PT_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE

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
		render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = ReSTIRPTReservoir();

		return;
	}

	// Initializing the random generator
	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(center_pixel_index));

	ReSTIRPTReservoir* input_reservoir_buffer = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	// Surface data of the center pixel
	ReSTIRSurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index, random_number_generator);

	int neighbors_confidence_sum_int = 0;
	unsigned int reuse_cell_index	 = spmis_get_reuse_cell_index(render_data, center_pixel_index, center_pixel_coords, center_pixel_surface,
																  neighbors_confidence_sum_int, random_number_generator);
	if (reuse_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		// Can happen if we completely fail to resolve hash collision and the pixel couldn't find a hash cell index. No spatial reuse in this case then
		ReSTIRPTReservoir center_pixel_reservoir														  = input_reservoir_buffer[center_pixel_index];
		render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = center_pixel_reservoir;

		return;
	}

	unsigned int reuse_cell_pixel_count =
		render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings.cell_pixels_counters[reuse_cell_index];
	int reused_neighbors_count = render_data.render_settings.restir_pt_settings.common_spatial_pass.reuse_neighbor_count;
	// Scaling the confidence sum, section 4.3 of the paper
	float non_canonical_confidence_scaling =
		render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings.do_non_canonical_confidence_adjustement
			? reused_neighbors_count / (float)reuse_cell_pixel_count
			: 1.0f;
	float neighbors_confidence_sum = neighbors_confidence_sum_int * non_canonical_confidence_scaling;

	ReSTIRPTReservoir spatial_reuse_output_reservoir;
	ReSTIRPTSpatialResamplingMISWeight<ReSTIR_PT_MISWeightsType> mis_weight_function;

	int center_pixel_reservoir_confidence = input_reservoir_buffer[center_pixel_index].M;
	if (render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings.cell_non_zero_reservoir_counters[reuse_cell_index] > 0)
	{
		// Resampling only the neighbors, canonical resampling is further below
		for (int neighbor_index = 0; neighbor_index < reused_neighbors_count; neighbor_index++)
		{
			float neighbor_selection_probability = 1.0f;
			unsigned int neighbor_pixel_index =
				get_spmis_spatial_neighbor_pixel_index(render_data, reuse_cell_index, neighbor_selection_probability, random_number_generator);
			if (neighbor_pixel_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				// Invalid neighbor
				continue;

			ReSTIRPTReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];

			float shift_mapping_jacobian = 1.0f;
			if (neighbor_reservoir.UCW > 0.0f && !neighbor_reservoir.sample.is_envmap_path())
			{
				// Only attempting the shift if the neighbor reservoir is valid
				//
				// Also, if this is the last neighbor resample (meaning that it is the center pixel),
				// the shift mapping is going to be an identity shift with a jacobian of 1 so we don't need to do it
				shift_mapping_jacobian = get_jacobian_determinant_reconnection_shift(
					neighbor_reservoir.sample.rc_vertex, neighbor_reservoir.sample.rc_vertex_geometric_normal.unpack(), center_pixel_surface.shading_point,
					render_data.g_buffer.primary_hit_position[neighbor_pixel_index],
					render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());
			}

			float target_function_at_center = 0.0f;
			if (neighbor_reservoir.UCW > 0.0f)
				target_function_at_center = ReSTIR_PT_evaluate_target_function<ReSTIR_PT_SpatialTargetFunctionVisibility>(
					render_data, neighbor_reservoir.sample, center_pixel_surface, random_number_generator);

			float mis_weight = mis_weight_function.get_resampling_MIS_weight_non_canonical(
				neighbor_reservoir.M * non_canonical_confidence_scaling, neighbor_reservoir.sample.target_function / shift_mapping_jacobian,
				center_pixel_reservoir_confidence,

				target_function_at_center, neighbors_confidence_sum, reused_neighbors_count, neighbor_selection_probability);

			spatial_reuse_output_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, shift_mapping_jacobian,
														random_number_generator);
			spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);
		}
	}

	// Now resampling the center pixel reservoir
	ReSTIRPTReservoir center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];
	if (center_pixel_reservoir.UCW > 0.0f)
	{
		// Sampling one random neighbor with uniform selection over all reservoirs, zero importance or not, (N_c = 1, section 4.2 of "Stochastic Pairwise MIS
		// for Unbiased Large - Kernel Reuse in Real - Time, Hedstrom et al. 2026") and using that neighbor to estimate the MIS weight of the canonical sample
		ReSTIRCommonSPMISSettings spmis_settings = ReSTIRSettingsHelper::get_restir_spmis_settings<ReSTIR_VARIANT_PT>(render_data);
		unsigned int cell_start_index			 = spmis_settings.cell_offsets[reuse_cell_index];
		unsigned int random_index				 = random_number_generator.random_index(reuse_cell_pixel_count);
		unsigned int neighbor_pixel_index		 = spmis_settings.pixel_indices_sorted[cell_start_index + random_index];
		float neighbor_selection_probability	 = 1.0f / reuse_cell_pixel_count;

		float shift_mapping_jacobian		 = 1.0f;
		float target_function_at_center		 = center_pixel_reservoir.sample.target_function;
		ReSTIRPTReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];

		float mis_weight =
			mis_weight_function.get_resampling_MIS_weight_canonical(render_data,

																	neighbor_reservoir.M * non_canonical_confidence_scaling, center_pixel_reservoir.sample,
																	center_pixel_reservoir.M, center_pixel_reservoir.sample.target_function,

																	center_pixel_surface, neighbor_pixel_index, neighbors_confidence_sum,
																	reused_neighbors_count, neighbor_selection_probability, random_number_generator);

		spatial_reuse_output_reservoir.combine_with(center_pixel_reservoir, mis_weight, target_function_at_center, shift_mapping_jacobian,
													random_number_generator);
		spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);
	}

	spatial_reuse_output_reservoir.M = reused_neighbors_count + center_pixel_reservoir.M;
	spatial_reuse_output_reservoir.end_with_normalization(1.0f, 1.0f);
	spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);

	// Validating that the sample point resampled is visible from our visible point
	// TODO use a flag in the sample reservoir to indicate whether we are unoccluded or not
	//		(we are always unoccluded if we resampled the canonical sample for example, in which case we don't have to do the validation)
	//		It would also probably be beneficial to have another kernel do the validation such that samples that don't need the validation
	//		(resampled the canonical neighbor) don't do the validation at all and we save on divergence
	ReSTIR_PT_visibility_validation(render_data, spatial_reuse_output_reservoir, center_pixel_surface.shading_point, center_pixel_surface.primitive_index,
									random_number_generator);

	// M-capping so that we don't have to M-cap when reading reservoirs on the next frame
	bool last_spatial_pass = render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_pass_index ==
							 render_data.render_settings.restir_pt_settings.common_spatial_pass.number_of_passes - 1;
	bool m_cap_enabled = render_data.render_settings.restir_pt_settings.m_cap > 0;
	if (last_spatial_pass && m_cap_enabled)
		// M-capping the spatial neighbor if an M-cap has been given
		spatial_reuse_output_reservoir.M = hippt::min(spatial_reuse_output_reservoir.M, render_data.render_settings.restir_pt_settings.m_cap);

	render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = spatial_reuse_output_reservoir;
	render_data.store_updated_random_seed(center_pixel_index, random_number_generator.m_state.seed);

#endif // ReSTIR_PT_MISWeightsType
}

#endif
