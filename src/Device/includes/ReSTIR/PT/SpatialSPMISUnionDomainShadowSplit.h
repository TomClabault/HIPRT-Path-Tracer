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
 * value of its probe without visibility is non-zero: same condition as the single kernel with the MIS target function BSDF approximation (except for a zero
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
#include "Device/includes/ReSTIR/PT/Reservoir.h"

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

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_H
