/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_H

// The shadow rays of this kernel are mostly visible and are not faster with the shadow ray traversal hint (Intersect.h): they keep the
// default hint. Defined before the includes so that Intersect.h sees it. GPU only: the CPU renderer compiles every kernel in one translation unit
#ifdef __KERNELCC__
#define SHADOW_RAY_TRAVERSAL_HINT hiprtTraversalHintDefault
#endif

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/SpatialGroupwiseMISWeight.h"
#include "Device/includes/ReSTIR/PT/SpatialMISTargetFunctionBSDFApproximation.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomain.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/PT/Utils.h"
#include "Device/includes/ReSTIR/PT/UtilsSpatialSPMIS.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Spatial reuse of the SPMIS union domain reuse (ReSTIRPTSPMISUnionDomainSettings, see SpatialSPMISUnionDomain.h) for the center pixel, used instead of the
 * legacy single reuse cell path when ReSTIRPTSettings::do_spmis_union_domain_reuse() is true (the render pass then launches the kernel of this file
 * instead of ReSTIR_PT_SpatialReuseSPMIS). Writes the output reservoir and the random seed of the pixel.
 *
 * Every resampled sample is visible from the center pixel: the neighbors are selected among the candidates whose visibility test passed and the
 * canonical sample is a path traced from the center pixel. The MIS weights evaluate the target functions of the other pixels with their visibility.
 * The output reservoir needs no visibility validation, whatever ReSTIR_PT_SpatialTargetFunctionVisibility is
 */
HIPRT_DEVICE void ReSTIR_PT_SPMIS_union_domain_spatial_reuse(HIPRTRenderData& render_data,
															 unsigned int center_pixel_index,
															 int2_t center_pixel_coords,
															 ReSTIRPTSurface& center_pixel_surface,
															 Xorshift32Generator& random_number_generator)
{
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings = render_data.render_settings.restir_pt_settings.spmis_settings.union_domain;
	ReSTIRPTReservoir* input_reservoir_buffer					  = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;

	ReSTIRPTSPMISUnionDomain domain =
		spmis_get_union_domain(render_data, center_pixel_index, center_pixel_coords, center_pixel_surface, random_number_generator);
	if (domain.cell_count == 0)
	{
		// Can happen if we completely fail to resolve hash collision and the pixel couldn't find a hash cell index. No spatial reuse in this case then
		ReSTIRPTReservoir center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];

		render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = center_pixel_reservoir;

		return;
	}

	int reused_neighbors_count = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS, union_domain_settings.neighbor_count);
	int mis_group_size		   = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_MIS_GROUP_SIZE, union_domain_settings.mis_group_size);
	int canonical_group_count  = hippt::max(1, union_domain_settings.canonical_group_count);

	ReSTIRPTReservoir spatial_reuse_output_reservoir;
	ReSTIRPTSpatialGroupwiseMISWeight mis_weight_function;

	int center_pixel_reservoir_confidence = input_reservoir_buffer[center_pixel_index].confidence;

	// For every neighbor slot, the selected pixel and its selection "probability" P (1 / unbiased contribution weight of the two-stage selection)
	unsigned int selected_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS];
	float selection_probabilities[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_NEIGHBORS];
	int selected_slot_count = 0;
	if (domain.non_zero_count > 0)
		// One pool of visibility candidates for the pixel
		selected_slot_count = spmis_union_domain_select_neighbors(render_data, center_pixel_surface, domain, center_pixel_reservoir_confidence,
																  reused_neighbors_count, union_domain_settings.visibility_pool_size, selected_pixel_indices,
																  selection_probabilities, random_number_generator);

	// The MIS weights use the MIS target function BSDF approximation instead of the target function (probes and own terms), the resampling weights use
	// the approximation times the luminance of the radiance of the sample (ReSTIR_PT_resampling_target_function_bsdf_approximation())

	// Resampling only the neighbors, canonical resampling is further below
	for (int neighbor_index = 0; neighbor_index < selected_slot_count; neighbor_index++)
	{
		unsigned int neighbor_pixel_index = selected_pixel_indices[neighbor_index];
		if (neighbor_pixel_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// All the candidates of this neighbor slot were occluded, no contribution
			continue;

		float neighbor_selection_probability = selection_probabilities[neighbor_index];
		ReSTIRPTReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];

		float shift_mapping_jacobian = 1.0f;
		if (neighbor_reservoir.UCW > 0.0f && !neighbor_reservoir.sample.is_envmap_path())
			// Only attempting the shift if the neighbor reservoir is valid
			shift_mapping_jacobian =
				get_jacobian_determinant_reconnection_shift(neighbor_reservoir.sample.rc_vertex, neighbor_reservoir.sample.rc_vertex_geometric_normal.unpack(),
															center_pixel_surface.shading_point, render_data.g_buffer.primary_hit_position[neighbor_pixel_index],
															render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

		// The selection traced the visibility of the sample at the center pixel: the visibility-free MIS approximation of the sample at the center pixel.
		// It is the own term of the sample at the center pixel in the MIS weights below (the probes and the own terms are the same function of (sample,
		// pixel)) and the resampling target function derives from it: the full BSDFs are only evaluated for the pairs that take the fallback
		float mis_target_at_center = 0.0f;
		if (neighbor_reservoir.UCW > 0.0f)
			mis_target_at_center = ReSTIR_PT_evaluate_target_function_at_pixel<false, true>(
				render_data, neighbor_reservoir.sample, static_cast<int>(center_pixel_index), center_pixel_surface.shading_point, random_number_generator);
		float resampling_target_function_at_center = ReSTIR_PT_resampling_target_function_bsdf_approximation(neighbor_reservoir.sample, mis_target_at_center);

		float mis_weight = 0.0f;
		if (resampling_target_function_at_center > 0.0f && shift_mapping_jacobian != 0.0f)
		{
			// Own term of the neighbor's sample at the neighbor in the measure of the center pixel, from its target function in its own measure (the Jacobian
			// division does not change the support)
			float mis_target_at_neighbor =
				ReSTIR_PT_MIS_target_function_bsdf_approximation_own_term(render_data, neighbor_reservoir.sample, static_cast<int>(neighbor_pixel_index),
																		  neighbor_reservoir.sample.target_function) /
				shift_mapping_jacobian;

			mis_weight = mis_weight_function.get_resampling_MIS_weight_non_canonical(
				render_data, neighbor_reservoir.sample, static_cast<float>(neighbor_reservoir.confidence), mis_target_at_neighbor,
				center_pixel_reservoir_confidence, mis_target_at_center, center_pixel_surface, domain, mis_group_size, reused_neighbors_count,
				neighbor_selection_probability, random_number_generator);
		}

		spatial_reuse_output_reservoir.combine_with(neighbor_reservoir, mis_weight, resampling_target_function_at_center, shift_mapping_jacobian,
													random_number_generator);
		spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);
	}

	// Now resampling the center pixel reservoir
	ReSTIRPTReservoir center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];
	if (center_pixel_reservoir.UCW > 0.0f)
	{
		// Own term of the canonical sample at the center pixel
		float canonical_mis_target = ReSTIR_PT_MIS_target_function_bsdf_approximation_own_term(
			render_data, center_pixel_reservoir.sample, static_cast<int>(center_pixel_index), center_pixel_reservoir.sample.target_function);

		float mis_weight = mis_weight_function.get_resampling_MIS_weight_canonical(
			render_data, center_pixel_reservoir.sample, center_pixel_reservoir.confidence, canonical_mis_target, center_pixel_surface, domain, mis_group_size,
			canonical_group_count, random_number_generator);

		// The resampling target function of the canonical sample is the one of its own term
		constexpr float shift_mapping_jacobian = 1.0f;
		float canonical_resampling_target_function =
			ReSTIR_PT_resampling_target_function_bsdf_approximation(center_pixel_reservoir.sample, canonical_mis_target);
		spatial_reuse_output_reservoir.combine_with(center_pixel_reservoir, mis_weight, canonical_resampling_target_function, shift_mapping_jacobian,
													random_number_generator);
		spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);
	}

	spatial_reuse_output_reservoir.confidence = reused_neighbors_count + center_pixel_reservoir.confidence;
	spatial_reuse_output_reservoir.end_with_normalization(1.0f, 1.0f);
	spatial_reuse_output_reservoir.sanity_check(center_pixel_coords);

	// M-capping so that we don't have to M-cap when reading reservoirs on the next frame. The union domain reuse only runs with a single spatial
	// reuse pass, this is the last one
	bool m_cap_enabled = render_data.render_settings.restir_pt_settings.m_cap > 0;
	if (m_cap_enabled)
		// M-capping the spatial neighbor if an M-cap has been given
		spatial_reuse_output_reservoir.confidence = hippt::min(spatial_reuse_output_reservoir.confidence, render_data.render_settings.restir_pt_settings.m_cap);

	render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = spatial_reuse_output_reservoir;
	render_data.store_updated_random_seed(center_pixel_index, random_number_generator.m_state.seed);
}

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_SpatialReuseSPMISUnionDomain()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SpatialReuseSPMISUnionDomain(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_RENDER_DATA);
#endif

	// Only compiling this whole kernel if we're using stochastic pairwise MIS: the union domain reuse is not used with the other MIS weights, see
	// ReSTIRPTSettings::do_spmis_union_domain_reuse()
#if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS

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

	// Surface data of the center pixel
	ReSTIRPTSurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index, random_number_generator);

	ReSTIR_PT_SPMIS_union_domain_spatial_reuse(render_data, center_pixel_index, center_pixel_coords, center_pixel_surface, random_number_generator);

#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_H
