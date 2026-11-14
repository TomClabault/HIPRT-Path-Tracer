/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_H
#define DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_H

/**
 * Shadow split version of the union domain spatial reuse ReSTIR_PT_SPMIS_union_domain_spatial_reuse() (SpatialReuseSPMISUnionDomain.h), used instead
 * of the single union domain kernel when ReSTIRPTSettings::use_spmis_union_domain_shadow_split_kernels() is true. GPU only.
 *
 * Why split the shadow rays out: the single kernel traces several shadow rays per pixel (stage 2 of the neighbor selection, the reverse probe of every chosen
 *neighbor, the canonical probes) in the middle of a lot of live state (domain, pool, surface, reservoir being built). That state stays in registers across
 * every BVH traversal, which lowers the occupancy of the kernel and makes each of its rays expensive. The shadow split moves the rays into two lean kernels
 * that keep almost nothing live during the traversal, the rest of the work stays in two heavy kernels that trace no ray:
 *
 *	- prepare (heavy, SpatialReuseSPMISUnionDomainShadowSplitPrepare.h): domain, stage 1 of the neighbor selection (the pool), the stage-2 ray record of
 *	  every pool pick, the uniform companion of every neighbor slot (the reverse probe pixel of the MIS weight of that slot), the canonical probes
 *	  (pixels, values without visibility, ray records), the duplicate map of the rays. The pixels without work (inactive, no primary hit, no domain)
 *	  are finished here exactly as the single kernel does
 *	- trace selection (lean, SpatialReuseSPMISUnionDomainShadowSplitTraceSelection.h): the stage-2 and canonical probe rays, one visibility bit each.
 *	  A ray that repeats the query of an earlier ray of the pixel is not traced, its bit is copied (see "Duplicate rays" below)
 *	- weights (heavy, SpatialReuseSPMISUnionDomainShadowSplitWeights.h): the choice of the neighbors among the visible picks (same partial Fisher-Yates
 *	  shuffle), the shifts, the target functions at the center, for every chosen neighbor its resampling weight with the companion visible and the
 *	  correction of that weight when the companion is occluded, the reverse probe ray records, the canonical MIS weight from the probe bits
 *	- resolve (lean, SpatialReuseSPMISUnionDomainShadowSplitResolve.h): the reverse probe rays, the final weights, the reservoir merge, the normalization,
 *	  the M-cap and the output
 *
 * The state that crosses kernels is in ReSTIRPTSPMISUnionDomainShadowSplitBuffers. The surface of the center pixel does not cross kernels: the weights kernel
 * rebuilds it.
 *
 * Same estimator as the single kernel: every random decision of the single kernel is made with the same probabilities, only the order in which the
 * random numbers of the pixel are drawn changes. The random numbers of a pixel come from ONE generator chain: the prepare kernel starts it from the
 * seed of the pixel and stores its state, the weights kernel continues from that state and stores it again, the resolve kernel draws the merge
 * uniforms from it and stores the random seed of the pixel (the only seed store, as in the single kernel). So:
 *	- the companions are drawn by the prepare kernel (uniform in the domain, independent of everything else, so they can be drawn before the
 *	  neighbors are chosen), which lets the weights kernel work without the domain
 *	- the Fisher-Yates shuffle draws its numbers in the weights kernel, after the visibility of the picks is known
 *	- the merge draws one uniform per candidate in the resolve kernel, after the visibility of the companions is known
 *	- the weights kernel rebuilds the surface of the center pixel from a SEPARATE generator started from the seed of the pixel (the seed buffer
 *	  is not written before the resolve kernel): its random decisions (the wavelength of a dispersive material) are then exactly those of the
 *	  prepare kernel, and the chain above is not disturbed
 *
 * The rays are the rays of the single kernel (same origin, endpoint, distance, origin primitive and alpha test seed). A probe ray is traced when the
 * value of its probe without visibility is non-zero: same condition as the single kernel with the MIS target approximation (except for a zero
 * Jacobian, which the single kernel tests after its ray), fewer rays without it (the single kernel then traces before evaluating the target
 * function). A ray whose value is zero cannot change the estimate. A ray that repeats the query of an earlier ray of the same pixel (same origin,
 * endpoint and seed) is traced once and its answer copied: the single kernel traces it again and gets the same answer.
 *
 * Every kernel runs one thread per pixel over the 8 x 8 blocks of the single kernel: the global BVH traversal stack has one slot per thread of
 * such a launch, a kernel with more threads than pixels would overflow it. The lean kernels distribute the rays of a block over its threads, see
 * spmis_union_domain_shadow_split_trace_rays_block()
 */

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/SpatialGroupwiseMISWeight.h"
#include "Device/includes/ReSTIR/PT/SpatialMISTargetApproximation.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomain.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

// Per-pixel flags written by the prepare kernel (ReSTIRPTSPMISUnionDomainShadowSplitBuffers::prepare_flags): bit p (p < 8) = pool pick p traces a stage-2
// ray, bit 8 + j = canonical probe j traces a ray, bit 16 + p = the sample of pool pick p is an envmap sample, then the single bits below
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_SELECTION_RAY_MASK	 = 0xFFFFu;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT	 = 8;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_MASK	 = 0xFFu;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PICK_ENVMAP_SHIFT	 = 16;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_ENVMAP_BIT = 1u << 24;
// The center pixel reservoir is non-zero: the canonical sample is resampled
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_PRESENT_BIT = 1u << 25;
// Stage 1 of the neighbor selection found a pool
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_POOL_VALID_BIT = 1u << 26;
// The prepare kernel finished the pixel (output written), the other kernels skip it
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT = 1u << 27;
// The pixel has duplicate rays: until the trace selection kernel, its visibility word holds its duplicate map (see "Duplicate rays" below)
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATES_BIT = 1u << 28;

// Duplicate map of a pixel, 32 bits:
//	- bits 3p .. 3p + 2, for every stage-2 pool pick p < 8: the pick whose ray answers the query of p
//	- bits 24 + 2j .. 25 + 2j, for every canonical probe j < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES: the probe whose
//	  ray answers the query of j (a probe index, not a ray bit)
// A pick or probe that is not a duplicate holds its own index. The canonical probes beyond the deduplicated ones are always traced
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_PICK_SOURCE_BITS		 = 3;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_SHIFT = 24;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_BITS	 = 2;
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES			 = 4;

// Per-pixel flags written by the weights kernel (ReSTIRPTSPMISUnionDomainShadowSplitBuffers::resolve_flags): bit k (k < 4) = neighbor slot k is a candidate
// of the merge, bit 4 + k = slot k traces a reverse probe ray, bit 8 + k = the sample of slot k is an envmap sample, then the single bits below
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_NEIGHBOR_SLOT_MASK	  = 0xFu;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_RAY_SHIFT	  = 4;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_ENVMAP_SHIFT  = 8;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_CANONICAL_BIT = 1u << 12;
static constexpr unsigned int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_DONE_BIT	  = 1u << 13;

static_assert(RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE <= 8, "The prepare flags hold one stage-2 ray bit and one envmap bit per pool pick");
static_assert(RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES <= 8, "The prepare flags hold one ray bit per canonical probe");
static_assert(RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MAX_NEIGHBORS <= 4, "The resolve flags hold three bits per neighbor slot");
static_assert(RESTIR_PT_SPMIS_UNION_DOMAIN_MAX_VISIBILITY_POOL_SIZE * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_PICK_SOURCE_BITS <=
				  RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_SHIFT,
			  "The pick fields of the duplicate map hold one pick index each, below the canonical probe fields");

// The shadow split kernels only run with an MIS group size of 2 (ReSTIRPTSettings::use_spmis_union_domain_shadow_split_kernels()): one companion per neighbor
static constexpr int RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE = 2;

/**
 * Duplicate rays of the trace selection kernel.
 *
 * A stage-2 ray is a function of the sample of the reservoir of its pool pick (reconnection vertex or envmap direction, alpha test seed of the
 * sample) and of the center pixel (origin and origin primitive: its primary hit), see spmis_union_domain_shadow_split_store_ray() and
 * spmis_union_domain_shadow_split_trace_pixel_ray(). Two picks of the same pool pixel read the same reservoir: their rays are the same query, whose
 * answer is deterministic (the alpha test generator is seeded by the ray record). Likewise, every canonical probe ray goes to the canonical sample
 * with its seed and starts at the primary hit of the probe pixel: two probes of the same domain pixel are the same query.
 *
 * The prepare kernel gives no ray bit to such a duplicate and records in the duplicate map of the pixel the earlier pick or probe whose ray answers
 * it: the first one of that pixel, which always has a ray bit itself. The trace selection kernel then traces a shorter list and copies the visibility
 * bit of every duplicate from its source. Every pick and probe stays its own term of the estimator (own selection probability, own probe value),
 * only the trace is shared. Duplicates are only searched among the picks and probes that need a ray: a pick or probe without a ray is never a
 * duplicate nor a source.
 *
 * The map travels from the prepare kernel to the trace selection kernel in the visibility word of the pixel (no extra buffer), which the trace
 * selection kernel then overwrites with the visibility bits
 */

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
 * Bit position in the duplicate map of the field of the ray 'ray_bit' (a ray bit of the prepare flags: a stage-2 pick, or a canonical probe below
 * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES)
 */
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_duplicate_field_shift(unsigned int ray_bit)
{
	if (ray_bit < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT)
		return ray_bit * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_PICK_SOURCE_BITS;

	unsigned int probe = ray_bit - RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT;

	return RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_SHIFT +
		   probe * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_BITS;
}

/**
 * Mask (in the low bits) of the field of the ray 'ray_bit' in the duplicate map
 */
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_duplicate_field_mask(unsigned int ray_bit)
{
	bool stage2_ray			= ray_bit < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT;
	unsigned int field_bits = stage2_ray ? RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_PICK_SOURCE_BITS
										 : RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATE_CANONICAL_SOURCE_BITS;

	return (1u << field_bits) - 1u;
}

/**
 * The ray bit whose visibility the ray 'ray_bit' copies according to the duplicate map (its own bit when it is not a duplicate)
 */
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_duplicate_source(unsigned int duplicate_map, unsigned int ray_bit)
{
	unsigned int field_shift  = spmis_union_domain_shadow_split_duplicate_field_shift(ray_bit);
	unsigned int source_index = (duplicate_map >> field_shift) & spmis_union_domain_shadow_split_duplicate_field_mask(ray_bit);

	bool stage2_ray = ray_bit < RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT;

	return stage2_ray ? source_index : RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + source_index;
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
 * The visibility bits 'visible_mask' of the traced rays of a pixel, completed with the bits of its duplicate rays copied from their sources (which
 * are never duplicates themselves)
 */
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_copy_duplicate_visibility(unsigned int visible_mask, unsigned int duplicate_map)
{
	unsigned int completed_visible_mask = visible_mask;

	unsigned int ray_bit_count =
		RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT + RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES;
	for (unsigned int ray_bit = 0; ray_bit < ray_bit_count; ray_bit++)
	{
		unsigned int source_ray_bit = spmis_union_domain_shadow_split_duplicate_source(duplicate_map, ray_bit);
		if (source_ray_bit == ray_bit)
			// Not a duplicate: traced, its bit is already there
			continue;

		unsigned int source_visible = (visible_mask >> source_ray_bit) & 1u;
		completed_visible_mask		= (completed_visible_mask & ~(1u << ray_bit)) | (source_visible << ray_bit);
	}

	return completed_visible_mask;
}

/**
 * Ray record 'entry_index' of 'rays': the reconnection vertex of the sample (or its envmap direction) and the bits of its alpha test seed
 */
HIPRT_DEVICE void spmis_union_domain_shadow_split_store_ray(float* rays, unsigned int entry_index, const ReSTIRPTReservoirSample& sample)
{
	reinterpret_cast<float4_t*>(rays)[entry_index] =
		make_float4(sample.rc_vertex.x, sample.rc_vertex.y, sample.rc_vertex.z, hippt::uint_as_float(sample.visible_to_sample_point_alpha_test_random_seed));
}

/**
 * The shadow ray of the single kernel from the primary hit of 'viewpoint_pixel' to the ray record 'ray' (reconnection vertex, or envmap direction when
 * 'envmap_sample'). Same origin, direction, distance, origin primitive and alpha test as ReSTIR_PT_evaluate_target_function<true>() at the surface of
 * the center pixel (whose shading point is its primary hit) and as ReSTIR_PT_evaluate_target_function_at_pixel<true>(). Returns true if visible
 */
HIPRT_DEVICE bool spmis_union_domain_shadow_split_trace_ray(const HIPRTRenderData& render_data,
															const float4_t& ray,
															bool envmap_sample,
															unsigned int viewpoint_pixel)
{
	float3_t endpoint			 = make_float3(ray.x, ray.y, ray.z);
	unsigned int alpha_test_seed = hippt::float_as_uint(ray.w);

	float3_t origin			   = render_data.g_buffer.primary_hit_position[viewpoint_pixel];
	int origin_primitive_index = render_data.g_buffer.first_hit_prim_index[viewpoint_pixel];

	float distance_to_sample_point;
	float3_t incident_light_direction;
	if (envmap_sample)
	{
		incident_light_direction = endpoint;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		incident_light_direction = endpoint - origin;
		distance_to_sample_point = hippt::length(incident_light_direction);
		incident_light_direction /= distance_to_sample_point;
	}

	hiprtRay visibility_ray;
	visibility_ray.origin	 = origin;
	visibility_ray.direction = incident_light_direction;

	Xorshift32Generator random_number_generator_alpha_test(alpha_test_seed);

	return !evaluate_shadow_ray_occluded(render_data, visibility_ray, distance_to_sample_point, origin_primitive_index, random_number_generator_alpha_test);
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
 * Prepare kernel for the center pixel: everything of the single kernel before its first shadow ray, see the top of this file
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
		float canonical_mis_target = center_pixel_reservoir.sample.target_function;
#if ReSTIR_PT_SPMISUnionDomainMISTargetApproximation == KERNEL_OPTION_TRUE
		canonical_mis_target = ReSTIR_PT_MIS_approximation_own_term(render_data, center_pixel_reservoir.sample, static_cast<int>(center_pixel_index),
																	center_pixel_reservoir.sample.target_function);
#endif

		shadow_split_buffers.canonical_numerators[center_pixel_index] = canonical_mis_target * center_pixel_reservoir_confidence;

		int probe_count =
			hippt::min(canonical_group_count * RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MIS_GROUP_SIZE, RESTIR_PT_SPMIS_UNION_DOMAIN_USTAT_MAX_PROBES);

		// Pixels of the deduplicated probes, only read for the probes that have a ray bit
		unsigned int probe_pixel_indices[RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DEDUPLICATED_CANONICAL_PROBES];

		for (int probe = 0; probe < probe_count; probe++)
		{
			unsigned int probe_pixel_index = spmis_union_domain_uniform_pixel(render_data, domain, random_number_generator);

			float probe_value = spmis_target_function_at_domain_pixel<false>(render_data, center_pixel_reservoir.sample, center_pixel_surface,
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
 * Traces the ray 'ray_bit' of the center pixel and returns true if it is visible. 'reverseRays' false: 'ray_bit' is a ray bit of the prepare flags
 * 'flags' (stage-2 pick or canonical probe). 'reverseRays' true: 'ray_bit' is a neighbor slot whose reverse probe ray the resolve flags 'flags' hold
 */
template <bool reverseRays>
HIPRT_DEVICE bool spmis_union_domain_shadow_split_trace_pixel_ray(const HIPRTRenderData& render_data,
																  unsigned int center_pixel_index,
																  unsigned int ray_bit,
																  unsigned int flags)
{
	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers =
		render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers;
	unsigned int pixel_count = shadow_split_buffers.pixel_count;

	float4_t ray;
	bool envmap_sample;
	unsigned int viewpoint_pixel;
	if constexpr (reverseRays)
	{
		// From the companion of the slot to the sample of the neighbor of the slot
		unsigned int slot_entry_index = ray_bit * pixel_count + center_pixel_index;

		ray				= reinterpret_cast<const float4_t*>(shadow_split_buffers.reverse_rays)[slot_entry_index];
		envmap_sample	= ((flags >> (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_ENVMAP_SHIFT + ray_bit)) & 1u) != 0;
		viewpoint_pixel = shadow_split_buffers.reverse_viewpoints[slot_entry_index];
	}
	else if (ray_bit >= RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT)
	{
		// From the probe pixel to the canonical sample
		unsigned int probe_entry_index = (ray_bit - RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_RAY_SHIFT) * pixel_count + center_pixel_index;

		ray				= reinterpret_cast<const float4_t*>(shadow_split_buffers.canonical_rays)[center_pixel_index];
		envmap_sample	= (flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_CANONICAL_ENVMAP_BIT) != 0;
		viewpoint_pixel = shadow_split_buffers.canonical_viewpoints[probe_entry_index];
	}
	else
	{
		// From the center pixel to the sample of the pool pick
		ray				= reinterpret_cast<const float4_t*>(shadow_split_buffers.selection_rays)[ray_bit * pixel_count + center_pixel_index];
		envmap_sample	= ((flags >> (RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PICK_ENVMAP_SHIFT + ray_bit)) & 1u) != 0;
		viewpoint_pixel = center_pixel_index;
	}

	return spmis_union_domain_shadow_split_trace_ray(render_data, ray, envmap_sample, viewpoint_pixel);
}

#ifdef __KERNELCC__
/**
 * Block-distributed tracing of the lean kernels: the rays of all the pixels of the 8 x 8 block form one list, pixel after pixel in thread order, and
 * thread t of the block traces the rays t, t + KernelWorkgroupThreadCount, ... of that list. 'ray_mask' is the set of rays of the pixel of the thread
 * (0 for a thread without pixel or without work).
 *
 * The number of rays varies a lot between pixels (from none to a dozen). Tracing them per pixel would keep the lanes of a pixel with few rays idle
 * while the other lanes of the wave trace theirs: here every lane has work while the list lasts, and consecutive lanes trace consecutive rays, mostly
 * rays of the same pixel, so from the same origin (coherent traversals). The list lives in shared memory, no global compaction is needed, and every
 * thread still owns one pixel, as the global traversal stack requires.
 *
 * Every thread of the block must call this function (barriers): no early return before it. Returns the visibility bits of the pixel of the thread
 * (same bit positions as its mask)
 */
template <bool reverseRays>
HIPRT_DEVICE unsigned int spmis_union_domain_shadow_split_trace_rays_block(const HIPRTRenderData& render_data,
																		   unsigned int center_pixel_index,
																		   unsigned int ray_mask,
																		   unsigned int flags)
{
	__shared__ unsigned int block_ray_masks[KernelWorkgroupThreadCount];
	__shared__ unsigned int block_flags[KernelWorkgroupThreadCount];
	__shared__ unsigned int block_pixels[KernelWorkgroupThreadCount];
	__shared__ unsigned int block_visible_masks[KernelWorkgroupThreadCount];
	// block_ray_offsets[t] = number of rays of the threads before t, block_ray_offsets[KernelWorkgroupThreadCount] = rays of the block
	__shared__ unsigned int block_ray_offsets[KernelWorkgroupThreadCount + 1];

	unsigned int thread_index = threadIdx.x + threadIdx.y * blockDim.x;

	block_ray_masks[thread_index]	  = ray_mask;
	block_flags[thread_index]		  = flags;
	block_pixels[thread_index]		  = center_pixel_index;
	block_visible_masks[thread_index] = 0;

	__syncthreads();

	// Exclusive prefix sum of the ray counts by one thread: a short serial loop, cheap next to the traversals
	if (thread_index == 0)
	{
		unsigned int ray_offset = 0;
		for (unsigned int thread = 0; thread < KernelWorkgroupThreadCount; thread++)
		{
			block_ray_offsets[thread] = ray_offset;
			ray_offset += hippt::popc(block_ray_masks[thread]);
		}

		block_ray_offsets[KernelWorkgroupThreadCount] = ray_offset;
	}

	__syncthreads();

	unsigned int block_ray_count = block_ray_offsets[KernelWorkgroupThreadCount];
	for (unsigned int block_ray = thread_index; block_ray < block_ray_count; block_ray += KernelWorkgroupThreadCount)
	{
		// Owner thread of the ray: block_ray_offsets[owner] <= block_ray < block_ray_offsets[owner + 1] (binary search, the owner has at least one ray)
		unsigned int low  = 0;
		unsigned int high = KernelWorkgroupThreadCount;
		while (high - low > 1)
		{
			unsigned int middle = (low + high) / 2;
			if (block_ray_offsets[middle] <= block_ray)
				low = middle;
			else
				high = middle;
		}

		unsigned int owner = low;

		// The ray is the (block_ray - offset)-th set bit of the mask of the owner
		unsigned int owner_mask = block_ray_masks[owner];
		for (unsigned int skipped = block_ray_offsets[owner]; skipped < block_ray; skipped++)
			owner_mask &= owner_mask - 1u;

		unsigned int ray_bit = hippt::lowest_set_bit(owner_mask);

		if (spmis_union_domain_shadow_split_trace_pixel_ray<reverseRays>(render_data, block_pixels[owner], ray_bit, block_flags[owner]))
			hippt::atomic_or(&block_visible_masks[owner], 1u << ray_bit);
	}

	__syncthreads();

	return block_visible_masks[thread_index];
}
#endif // #ifdef __KERNELCC__

/**
 * Weights kernel for the center pixel whose prepare flags are 'prepare_flags' (pixel not done) and whose stage-2 / canonical probe visibility bits are
 * 'visible_mask', see the top of this file
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
	// prepare kernel, see the top of this file
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
#if ReSTIR_PT_SPMISUnionDomainMISTargetApproximation == KERNEL_OPTION_TRUE
			mis_target_at_neighbor = ReSTIR_PT_MIS_approximation_own_term(render_data, neighbor_reservoir.sample, static_cast<int>(neighbor_pixel_index),
																		  neighbor_reservoir.sample.target_function) /
									 shift_mapping_jacobian;
			mis_target_at_center =
				ReSTIR_PT_MIS_approximation_own_term(render_data, neighbor_reservoir.sample, static_cast<int>(center_pixel_index), target_function_at_center);
#endif // #if ReSTIR_PT_SPMISUnionDomainMISTargetApproximation == KERNEL_OPTION_TRUE

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
 * Resolve kernel for the center pixel whose resolve flags are 'resolve_flags' (pixel not done) once its reverse probe rays are traced
 * ('companion_visible_mask': the slots whose companion is visible), see the top of this file.
 *
 * Merges the candidate slots in slot order then the canonical sample (the order of the single kernel) with streaming resampling. One uniform is drawn
 * per candidate whatever its weight (a zero weight is never selected): same distribution as ReSTIRPTReservoir::combine_with(), but only the sample
 * of the selected candidate is loaded, once, at the end. Then the normalization, the M-cap, the output reservoir and the random seed of the pixel
 */
HIPRT_DEVICE void spmis_union_domain_shadow_split_resolve_pixel(HIPRTRenderData& render_data,
																unsigned int center_pixel_index,
																unsigned int resolve_flags,
																unsigned int companion_visible_mask)
{
	const ReSTIRPTSPMISUnionDomainSettings& union_domain_settings		   = render_data.render_settings.restir_pt_settings.spmis_settings.union_domain;
	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers = union_domain_settings.shadow_split_buffers;
	ReSTIRPTReservoir* input_reservoir_buffer							   = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs;
	unsigned int pixel_count											   = shadow_split_buffers.pixel_count;

	// Continues the generator chain of the weights kernel
	Xorshift32Generator random_number_generator(shadow_split_buffers.random_states[center_pixel_index]);

	// Slots that traced a reverse probe ray whose companion is occluded
	unsigned int reverse_ray_mask =
		(resolve_flags >> RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_REVERSE_RAY_SHIFT) & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_NEIGHBOR_SLOT_MASK;
	unsigned int companion_occluded_mask = reverse_ray_mask & ~companion_visible_mask;

	// The canonical sample is the candidate after the neighbor slots
	constexpr int canonical_candidate = RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MAX_NEIGHBORS;

	float weight_sum = 0.0f;
	// -1 = nothing selected
	int selected_candidate = -1;

	unsigned int candidate_mask = resolve_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_NEIGHBOR_SLOT_MASK;
	while (candidate_mask != 0)
	{
		unsigned int slot = hippt::lowest_set_bit(candidate_mask);
		candidate_mask &= candidate_mask - 1u;

		// w = w0 + (1 - V) x correction
		unsigned int slot_entry_index = slot * pixel_count + center_pixel_index;
		float weight				  = shadow_split_buffers.neighbor_weights[slot_entry_index];
		if ((companion_occluded_mask >> slot) & 1u)
			weight += shadow_split_buffers.neighbor_weight_corrections[slot_entry_index];

		float acceptance_uniform = random_number_generator();
		if (!(weight > 0.0f))
			continue;

		weight_sum += weight;
		if (acceptance_uniform < weight / weight_sum)
			selected_candidate = static_cast<int>(slot);
	}

	if (resolve_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_RESOLVE_CANONICAL_BIT)
	{
		float weight			 = shadow_split_buffers.canonical_weights[center_pixel_index];
		float acceptance_uniform = random_number_generator();
		if (weight > 0.0f)
		{
			weight_sum += weight;
			if (acceptance_uniform < weight / weight_sum)
				selected_candidate = canonical_candidate;
		}
	}

	ReSTIRPTReservoir spatial_reuse_output_reservoir;
	if (selected_candidate == canonical_candidate)
		// The canonical sample keeps its target function
		spatial_reuse_output_reservoir.sample = input_reservoir_buffer[center_pixel_index].sample;
	else if (selected_candidate >= 0)
	{
		// The sample of the selected neighbor with its target function at the center pixel
		unsigned int slot_entry_index						  = selected_candidate * pixel_count + center_pixel_index;
		spatial_reuse_output_reservoir.sample				  = input_reservoir_buffer[shadow_split_buffers.neighbor_pixels[slot_entry_index]].sample;
		spatial_reuse_output_reservoir.sample.target_function = shadow_split_buffers.neighbor_targets[slot_entry_index];
	}
	spatial_reuse_output_reservoir.weight_sum = weight_sum;

	int reused_neighbors_count = hippt::clamp(1, RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_MAX_NEIGHBORS, union_domain_settings.neighbor_count);

	spatial_reuse_output_reservoir.confidence = reused_neighbors_count + input_reservoir_buffer[center_pixel_index].confidence;
	spatial_reuse_output_reservoir.end_with_normalization(1.0f, 1.0f);

	// M-capping as the single kernel
	bool m_cap_enabled = render_data.render_settings.restir_pt_settings.m_cap > 0;
	if (m_cap_enabled)
		spatial_reuse_output_reservoir.confidence = hippt::min(spatial_reuse_output_reservoir.confidence, render_data.render_settings.restir_pt_settings.m_cap);

	render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs[center_pixel_index] = spatial_reuse_output_reservoir;
	render_data.store_updated_random_seed(center_pixel_index, random_number_generator.m_state.seed);
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_H
