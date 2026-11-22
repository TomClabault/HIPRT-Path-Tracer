/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PT/SpatialGroupwiseMISWeight.h"
#include "Device/includes/ReSTIR/PT/SpatialMISTargetFunctionBSDFApproximation.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomain.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomainShadowSplit.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * The duplicate map in which no ray is a duplicate: every field holds its own index
 */
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_duplicate_map_identity()
{
	unsigned int duplicate_map = 0;

	for (unsigned int pick = 0; pick < RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE; pick++)
		duplicate_map |= pick << (pick * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_PICK_SOURCE_BITS);

	for (unsigned int probe = 0; probe < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES; probe++)
		duplicate_map |= probe << (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_SHIFT +
								   probe * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_BITS);

	return duplicate_map;
}

/**
 * 'duplicate_map' with the ray 'ray_bit' marked as a duplicate of the earlier ray 'source_ray_bit' of the same class (stage-2 pick or canonical
 * probe)
 */
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_set_duplicate_source(unsigned int duplicate_map, unsigned int ray_bit, unsigned int source_ray_bit)
{
	bool stage2_ray			  = ray_bit < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT;
	unsigned int source_index = stage2_ray ? source_ray_bit : source_ray_bit - RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT;

	unsigned int field_shift = spmis_union_domain_shadow_split_duplicate_field_shift(ray_bit);
	unsigned int field_mask	 = spmis_union_domain_shadow_split_duplicate_field_mask(ray_bit) << field_shift;

	return (duplicate_map & ~field_mask) | (source_index << field_shift);
}

/**
 * True when ReSTIR_PT_sample_visible_at_surface() (stage 2 of spmis_union_domain_select_neighbors()) traces its shadow ray for this sample at this
 * surface. These are the zero tests of ReSTIR_PT_evaluate_target_function() (degenerate distance, backfacing light of a DI sample, cosine <= 0 outside of
 * a transmission lobe), duplicated here because that function traces the ray right after them: keep both in sync
 */
HIPRT_DEVICE bool spmis_union_domain_shadow_split_stage2_needs_ray(const ReSTIRPTReservoirSample& sample, const ReSTIRPTSurface& surface)
{
	float3_t incident_light_direction;
	if (sample.is_envmap_path())
		incident_light_direction = sample.rc_vertex;
	else
	{
		incident_light_direction	   = sample.rc_vertex - surface.shading_point;
		float distance_to_sample_point = hippt::length(incident_light_direction);
		if (distance_to_sample_point <= 1.0e-6f)
			return false;

		incident_light_direction /= distance_to_sample_point;
	}

	if (!sample.is_envmap_path() && sample.di_sample &&
		compute_cosine_term_at_light_source(sample.rc_vertex_geometric_normal.unpack(), -incident_light_direction) <= 0.0f)
		return false;

	float cosine_term = hippt::dot(incident_light_direction, surface.shading_normal);
	if (cosine_term <= 0.0f && !bsdf_incident_light_info_transmission_lobe(sample.incident_light_info_at_visible_point))
		return false;

	return true;
}

/**
 * Prepare kernel for the center pixel: everything of the single kernel before its first shadow ray, see the top of SpatialSPMISUnionDomainShadowSplit.h
 */
HIPRT_DEVICE void spmis_union_domain_shadow_split_prepare_pixel(HIPRTRenderData& render_data, unsigned int center_pixel_index, int2_t center_pixel_coords)
{
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings		   = render_data.render_settings.restir_pt_settings.spmis_settings.union_domain;
	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers = union_domain_settings.shadow_split_buffers;
	ReSTIRPTReservoir* input_reservoir_buffer							   = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	unsigned int pixel_count											   = shadow_split_buffers.pixel_count;

	if (!render_data.aux_buffers.pixel_active[center_pixel_index] || render_data.g_buffer.first_hit_prim_index[center_pixel_index] == -1)
	{
		// As the single kernel: pixel inactive because of adaptive sampling or no primary hit, empty output reservoir, no random seed stored
		render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = ReSTIRPTReservoir();
		shadow_split_buffers.prepare_flags[center_pixel_index] = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT;
		shadow_split_buffers.resolve_flags[center_pixel_index] = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_DONE_BIT;

		return;
	}

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(center_pixel_index));
	ReSTIRPTSurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index, random_number_generator);

	ReSTIRPTSPMISUnionDomain domain =
		spmis_get_union_domain(render_data, center_pixel_index, center_pixel_coords, center_pixel_surface, random_number_generator);
	if (domain.cell_count == 0)
	{
		// As the single kernel: no spatial reuse, the input reservoir is the output, no random seed stored
		render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = input_reservoir_buffer[center_pixel_index];
		shadow_split_buffers.prepare_flags[center_pixel_index] = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT;
		shadow_split_buffers.resolve_flags[center_pixel_index] = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_DONE_BIT;

		return;
	}

	int reused_neighbors_count = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MAX_NEIGHBORS, union_domain_settings.neighbor_count);
	int canonical_group_count  = hippt::max(1, union_domain_settings.canonical_group_count);
	int pool_size			   = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE, union_domain_settings.visibility_pool_size);

	int center_pixel_reservoir_confidence = input_reservoir_buffer[center_pixel_index].confidence;

	unsigned int prepare_flags = 0;
	// Duplicate rays of the pixel, see spmis_union_domain_shadow_split_duplicate_map_identity()
	unsigned int duplicate_map = spmis_union_domain_shadow_split_duplicate_map_identity();

	// Stage 1 of spmis_union_domain_select_neighbors() and the stage-2 ray records of the picks that trace one
	if (domain.non_zero_count > 0)
	{
		unsigned int pool_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
		float pool_selection_probabilities[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];
		float pool_selection_targets[RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE];

		bool pool_valid =
			spmis_union_domain_select_pool_candidates(render_data, center_pixel_surface, domain, center_pixel_reservoir_confidence, pool_size,
													  pool_pixel_indices, pool_selection_probabilities, pool_selection_targets, random_number_generator);
		if (pool_valid)
		{
			prepare_flags |= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_POOL_VALID_BIT;

			for (int pick = 0; pick < pool_size; pick++)
			{
				// Same skipped picks as stage 2 of spmis_union_domain_select_neighbors()
				if (pool_selection_targets[pick] <= 0.0f || pool_selection_probabilities[pick] <= 0.0f)
					continue;

				const ReSTIRPTReservoir& pick_reservoir = input_reservoir_buffer[pool_pixel_indices[pick]];
				if (pick_reservoir.UCW <= 0.0f)
					continue;

				if (!spmis_union_domain_shadow_split_stage2_needs_ray(pick_reservoir.sample, center_pixel_surface))
					// No ray: not visible for the selection
					continue;

				unsigned int entry_index = pick * pixel_count + center_pixel_index;

				// A duplicate still is its own pick of the selection
				shadow_split_buffers.pick_pixels[entry_index]		 = pool_pixel_indices[pick];
				shadow_split_buffers.pick_probabilities[entry_index] = pool_selection_probabilities[pick];

				// An earlier pick of the same pool pixel with a ray bit is the same query. The first pick of a pool pixel has a ray bit, the later ones
				// are its duplicates
				int source_pick = pick;
				for (int earlier_pick = 0; earlier_pick < pick; earlier_pick++)
				{
					bool earlier_pick_ray = (prepare_flags >> earlier_pick) & 1u;
					if (earlier_pick_ray && pool_pixel_indices[earlier_pick] == pool_pixel_indices[pick])
					{
						source_pick = earlier_pick;

						break;
					}
				}

				if (source_pick != pick)
				{
					// No ray: the trace selection kernel copies the visibility bit of the source pick
					duplicate_map = spmis_union_domain_shadow_split_set_duplicate_source(duplicate_map, pick, source_pick);
					prepare_flags |= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATES_BIT;

					continue;
				}

				spmis_union_domain_shadow_split_store_ray(shadow_split_buffers.selection_rays, entry_index, pick_reservoir.sample);

				prepare_flags |= 1u << pick;
				if (pick_reservoir.sample.is_envmap_path())
					prepare_flags |= 1u << (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PICK_ENVMAP_SHIFT + pick);
			}

			// The uniform companion of every neighbor slot (the g - 1 = 1 pixel of the group of get_resampling_MIS_weight_non_canonical()), drawn here
			// uniformly in the domain and independently of the selection
			for (int slot = 0; slot < reused_neighbors_count; slot++)
				shadow_split_buffers.companions[slot * pixel_count + center_pixel_index] =
					spmis_union_domain_uniform_pixel(render_data, domain, random_number_generator);
		}
	}

	// Canonical probes of get_resampling_MIS_weight_canonical(): pixels and values without visibility here, rays in the trace selection kernel, MIS
	// weight in the weights kernel
	const ReSTIRPTReservoir& center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];
	if (center_pixel_reservoir.UCW > 0.0f)
	{
		prepare_flags |= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_PRESENT_BIT;

		// Own term of the canonical sample at the center pixel
		float canonical_mis_target = ReSTIR_PT_MIS_target_function_bsdf_approximation_own_term(
			render_data, center_pixel_reservoir.sample, static_cast<int>(center_pixel_index), center_pixel_reservoir.sample.target_function);

		shadow_split_buffers.canonical_numerators[center_pixel_index] = canonical_mis_target * center_pixel_reservoir_confidence;

		int probe_count =
			hippt::min(canonical_group_count * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE, RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES);

		// Pixels of the deduplicated probes, only read for the probes that have a ray bit
		unsigned int probe_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES];

		for (int probe = 0; probe < probe_count; probe++)
		{
			unsigned int probe_pixel_index = spmis_union_domain_uniform_pixel(render_data, domain, random_number_generator);

			float probe_value = spmis_target_function_at_domain_pixel<false>(render_data, center_pixel_reservoir.sample, center_pixel_surface.shading_point,
																			 probe_pixel_index, random_number_generator);
			if (probe_value > 0.0f)
			{
				// Only the probes with a ray can be non-zero: the weights kernel reads the value of a probe whose ray is visible only. A duplicate keeps
				// its own value, only its visibility is copied
				unsigned int entry_index = probe * pixel_count + center_pixel_index;

				shadow_split_buffers.canonical_values[entry_index] = probe_value;

				// An earlier probe of the same domain pixel with a ray bit is the same query
				int source_probe = probe;
				if (probe < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES)
				{
					for (int earlier_probe = 0; earlier_probe < probe; earlier_probe++)
					{
						bool earlier_probe_ray = (prepare_flags >> (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + earlier_probe)) & 1u;
						if (earlier_probe_ray && probe_pixel_indices[earlier_probe] == probe_pixel_index)
						{
							source_probe = earlier_probe;

							break;
						}
					}

					probe_pixel_indices[probe] = probe_pixel_index;
				}

				if (source_probe != probe)
				{
					// No ray: the trace selection kernel copies the visibility bit of the source probe
					unsigned int probe_ray_bit		  = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + probe;
					unsigned int source_probe_ray_bit = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + source_probe;

					duplicate_map = spmis_union_domain_shadow_split_set_duplicate_source(duplicate_map, probe_ray_bit, source_probe_ray_bit);
					prepare_flags |= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATES_BIT;

					continue;
				}

				shadow_split_buffers.canonical_viewpoints[entry_index] = probe_pixel_index;

				prepare_flags |= 1u << (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + probe);
			}
		}

		// Every canonical probe ray goes to the canonical sample: one ray record for all of them
		if ((prepare_flags >> RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT) & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_MASK)
			spmis_union_domain_shadow_split_store_ray(shadow_split_buffers.canonical_rays, center_pixel_index, center_pixel_reservoir.sample);

		if (center_pixel_reservoir.sample.is_envmap_path())
			prepare_flags |= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_ENVMAP_BIT;
	}

	shadow_split_buffers.domain_confidence_sums[center_pixel_index] = static_cast<float>(domain.confidence_sum);
	shadow_split_buffers.prepare_flags[center_pixel_index]			= prepare_flags;
	shadow_split_buffers.random_states[center_pixel_index]			= random_number_generator.m_state.seed;

	// The duplicate map goes to the trace selection kernel through the visibility word, which that kernel overwrites with the visibility bits
	if (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATES_BIT)
		shadow_split_buffers.selection_visibility[center_pixel_index] = duplicate_map;
}

/**
 * Prepare kernel of the shadow split union domain spatial reuse (see SpatialSPMISUnionDomainShadowSplit.h): domain, pool, stage-2 and canonical probe
 * ray records. Traces no ray. One thread per pixel over the 8 x 8 blocks of the render resolution, as the single union domain kernel.
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitPrepare()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitPrepare(HIPRTRenderData render_data, int x, int y)
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

	unsigned int center_pixel_index = x + y * render_data.render_settings.render_resolution.x;

	spmis_union_domain_shadow_split_prepare_pixel(render_data, center_pixel_index, make_int2(x, y));

#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PREPARE_H
