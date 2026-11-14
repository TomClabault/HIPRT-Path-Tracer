/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/PT/SpatialGroupwiseMISWeight.h"
#include "Device/includes/ReSTIR/PT/SpatialMISTargetFunctionBSDFApproximation.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomain.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomainShadowSplit.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Weights kernel for the center pixel whose prepare flags are 'prepare_flags' (pixel not done) and whose stage-2 / canonical probe visibility bits are
 * 'visible_mask', see the top of SpatialSPMISUnionDomainShadowSplit.h
 */
HIPRT_DEVICE void spmis_union_domain_shadow_split_weights_pixel(HIPRTRenderData& render_data,
																unsigned int center_pixel_index,
																unsigned int prepare_flags,
																unsigned int visible_mask)
{
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings		   = render_data.render_settings.restir_pt_settings.spmis_settings.union_domain;
	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers = union_domain_settings.shadow_split_buffers;
	ReSTIRPTReservoir* input_reservoir_buffer							   = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	unsigned int pixel_count											   = shadow_split_buffers.pixel_count;

	// Continues the generator chain of the prepare kernel
	Xorshift32Generator random_number_generator(shadow_split_buffers.random_states[center_pixel_index]);

	// The surface of the prepare kernel, rebuilt with a separate generator started from the same seed so that its random decisions are those of the
	// prepare kernel, see the top of SpatialSPMISUnionDomainShadowSplit.h
	Xorshift32Generator surface_random_number_generator(render_data.get_updated_random_seed(center_pixel_index));
	ReSTIRPTSurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index, surface_random_number_generator);

	int reused_neighbors_count = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MAX_NEIGHBORS, union_domain_settings.neighbor_count);
	int canonical_group_count  = hippt::max(1, union_domain_settings.canonical_group_count);
	int pool_size			   = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE, union_domain_settings.visibility_pool_size);

	int center_pixel_reservoir_confidence = input_reservoir_buffer[center_pixel_index].confidence;
	float domain_confidence_sum			  = shadow_split_buffers.domain_confidence_sums[center_pixel_index];

	ReSTIRPTSpatialGroupwiseMISWeight mis_weight_function;

	unsigned int resolve_flags = 0;

	if (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_POOL_VALID_BIT)
	{
		// Stage 2 of spmis_union_domain_select_neighbors(): the visible picks in pick order, min(N, n_vis) of them chosen uniformly without replacement
		int visible_picks[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
		int visible_pick_count = 0;
		for (int pick = 0; pick < pool_size; pick++)
			if ((visible_mask >> pick) & 1u)
				visible_picks[visible_pick_count++] = pick;

		int chosen_count			= hippt::min(reused_neighbors_count, visible_pick_count);
		float inclusion_probability = visible_pick_count > reused_neighbors_count ? reused_neighbors_count / static_cast<float>(visible_pick_count) : 1.0f;

		for (int chosen = 0; chosen < chosen_count; chosen++)
		{
			int swapped_index			 = chosen + static_cast<int>(random_number_generator.random_index(visible_pick_count - chosen));
			int chosen_pick				 = visible_picks[swapped_index];
			visible_picks[swapped_index] = visible_picks[chosen];
			visible_picks[chosen]		 = chosen_pick;

			unsigned int pick_entry_index	  = chosen_pick * pixel_count + center_pixel_index;
			unsigned int neighbor_pixel_index = shadow_split_buffers.pick_pixels[pick_entry_index];
			float neighbor_selection_probability =
				inclusion_probability * pool_size * shadow_split_buffers.pick_probabilities[pick_entry_index] / reused_neighbors_count;

			// From here, the neighbor loop of ReSTIR_PT_SPMIS_union_domain_spatial_reuse()
			ReSTIRPTReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];

			float shift_mapping_jacobian = 1.0f;
			if (neighbor_reservoir.UCW > 0.0f && !neighbor_reservoir.sample.is_envmap_path())
				shift_mapping_jacobian = get_jacobian_determinant_reconnection_shift(
					neighbor_reservoir.sample.rc_vertex, neighbor_reservoir.sample.rc_vertex_geometric_normal.unpack(), center_pixel_surface.shading_point,
					render_data.g_buffer.primary_hit_position[neighbor_pixel_index],
					render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

			// The selection traced the visibility of the sample at the center pixel: the visibility-free target function is the target function
			float target_function_at_center = 0.0f;
			if (neighbor_reservoir.UCW > 0.0f)
				target_function_at_center =
					ReSTIR_PT_evaluate_target_function<false>(render_data, neighbor_reservoir.sample, center_pixel_surface, random_number_generator);

			if (!(target_function_at_center > 0.0f && shift_mapping_jacobian != 0.0f))
				// Zero MIS weight in the single kernel: zero resampling weight, never selected
				continue;

			// Target function of the neighbor's sample at the neighbor in the measure of the center pixel
			float target_function_at_neighbor = neighbor_reservoir.sample.target_function / shift_mapping_jacobian;

			float mis_target_at_neighbor = target_function_at_neighbor;
			float mis_target_at_center	 = target_function_at_center;
#if ReSTIR_PT_SPMISUnionDomainMISTargetFunctionBSDFApproximation == KERNEL_OPTION_TRUE
			mis_target_at_neighbor =
				ReSTIR_PT_MIS_target_function_bsdf_approximation_own_term(render_data, neighbor_reservoir.sample, static_cast<int>(neighbor_pixel_index),
																		  neighbor_reservoir.sample.target_function) /
				shift_mapping_jacobian;
			mis_target_at_center = ReSTIR_PT_MIS_target_function_bsdf_approximation_own_term(render_data, neighbor_reservoir.sample,
																							 static_cast<int>(center_pixel_index), target_function_at_center);
#endif // #if ReSTIR_PT_SPMISUnionDomainMISTargetFunctionBSDFApproximation == KERNEL_OPTION_TRUE

			// The companion of the slot drawn by the prepare kernel and its value without visibility: its ray is traced by the resolve kernel
			unsigned int companion_pixel_index = shadow_split_buffers.companions[chosen * pixel_count + center_pixel_index];
			float companion_target_function	   = spmis_target_function_at_domain_pixel<false>(render_data, neighbor_reservoir.sample, center_pixel_surface,
																							  companion_pixel_index, random_number_generator);
			bool reverse_ray_needed			   = companion_target_function > 0.0f;

			// The MIS weight of get_resampling_MIS_weight_non_canonical() for both visibilities V of the companion: the group sum is p_i + V q_J
			float mis_weight_companion_visible = mis_weight_function.get_MIS_weight_non_canonical_from_group_sum(
				static_cast<float>(neighbor_reservoir.confidence), mis_target_at_neighbor, center_pixel_reservoir_confidence, mis_target_at_center,
				domain_confidence_sum, RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE, reused_neighbors_count, neighbor_selection_probability,
				mis_target_at_neighbor + companion_target_function);

			float mis_weight_companion_occluded = mis_weight_companion_visible;
			if (reverse_ray_needed)
				mis_weight_companion_occluded = mis_weight_function.get_MIS_weight_non_canonical_from_group_sum(
					static_cast<float>(neighbor_reservoir.confidence), mis_target_at_neighbor, center_pixel_reservoir_confidence, mis_target_at_center,
					domain_confidence_sum, RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE, reused_neighbors_count, neighbor_selection_probability,
					mis_target_at_neighbor);

			// The resampling weight of ReSTIRPTReservoir::combine_with(): MIS weight x target function x UCW x Jacobian. The resolve kernel adds the
			// correction when the companion is occluded
			float resampling_factor			  = target_function_at_center * neighbor_reservoir.UCW * shift_mapping_jacobian;
			float weight_companion_visible	  = mis_weight_companion_visible * resampling_factor;
			float weight_occlusion_correction = (mis_weight_companion_occluded - mis_weight_companion_visible) * resampling_factor;

			// Writeback of the slot: reverse probe ray, then the merge candidate
			unsigned int slot_entry_index = chosen * pixel_count + center_pixel_index;

			if (reverse_ray_needed)
			{
				spmis_union_domain_shadow_split_store_ray(shadow_split_buffers.reverse_rays, slot_entry_index, neighbor_reservoir.sample);
				shadow_split_buffers.reverse_viewpoints[slot_entry_index] = companion_pixel_index;

				resolve_flags |= 1u << (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_RAY_SHIFT + chosen);
				if (neighbor_reservoir.sample.is_envmap_path())
					resolve_flags |= 1u << (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_ENVMAP_SHIFT + chosen);
			}

			if (weight_companion_visible > 0.0f || reverse_ray_needed)
			{
				shadow_split_buffers.neighbor_pixels[slot_entry_index]			   = neighbor_pixel_index;
				shadow_split_buffers.neighbor_targets[slot_entry_index]			   = target_function_at_center;
				shadow_split_buffers.neighbor_weights[slot_entry_index]			   = weight_companion_visible;
				shadow_split_buffers.neighbor_weight_corrections[slot_entry_index] = weight_occlusion_correction;

				resolve_flags |= 1u << chosen;
			}
		}
	}

	// Canonical MIS weight of get_resampling_MIS_weight_canonical() from the probe values and their visibility bits (a probe without a ray is zero)
	if (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_PRESENT_BIT)
	{
		float canonical_numerator = shadow_split_buffers.canonical_numerators[center_pixel_index];
		// The g pixels of a group stand for the whole domain
		float group_scale = domain_confidence_sum / RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE;

		int probe_count =
			hippt::min(canonical_group_count * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE, RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES);

		float probe_target_functions[RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES];
		for (int probe = 0; probe < probe_count; probe++)
		{
			bool probe_visible			  = (visible_mask >> (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + probe)) & 1u;
			probe_target_functions[probe] = probe_visible ? shadow_split_buffers.canonical_values[probe * pixel_count + center_pixel_index] : 0.0f;
		}

		float canonical_mis_weight = mis_weight_function.get_canonical_MIS_weight_from_probes(
			canonical_numerator, group_scale, probe_target_functions, probe_count, RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE);

		// Resampling weight of the canonical sample: MIS weight x its target function x its UCW (Jacobian 1)
		const ReSTIRPTReservoir& center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];
		shadow_split_buffers.canonical_weights[center_pixel_index] =
			canonical_mis_weight * center_pixel_reservoir.sample.target_function * center_pixel_reservoir.UCW;

		resolve_flags |= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_CANONICAL_BIT;
	}

	shadow_split_buffers.resolve_flags[center_pixel_index] = resolve_flags;
	shadow_split_buffers.random_states[center_pixel_index] = random_number_generator.m_state.seed;
}

/**
 * Weights kernel of the shadow split union domain spatial reuse (see SpatialSPMISUnionDomainShadowSplit.h): choice of the neighbors among the visible
 * picks, shifts, resampling weights for both visibilities of the companions, reverse probe ray records, canonical MIS weight. Traces no ray. One thread
 * per pixel over the 8 x 8 blocks of the render resolution, as the single union domain kernel.
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitWeights()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitWeights(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_RENDER_DATA);
#endif

	// Only compiling this whole kernel if we're using stochastic pairwise MIS: the union domain reuse is not used with the other MIS weights
#if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS

#ifdef __KERNELCC__
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers =
		render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers;

	unsigned int center_pixel_index = x + y * render_data.render_settings.render_resolution.x;

	unsigned int prepare_flags = shadow_split_buffers.prepare_flags[center_pixel_index];
	if (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT)
		// Finished by the prepare kernel
		return;

	spmis_union_domain_shadow_split_weights_pixel(render_data, center_pixel_index, prepare_flags,
												  shadow_split_buffers.selection_visibility[center_pixel_index]);

#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_H
