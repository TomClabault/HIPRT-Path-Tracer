/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_H

// The shadow rays of this kernel are mostly visible and are not faster with the shadow ray traversal hint (Intersect.h): they keep the default hint.
// Defined before the includes so that Intersect.h sees it. GPU only: the CPU renderer compiles every kernel in one translation unit
#ifdef __KERNELCC__
#define SHADOW_RAY_TRAVERSAL_HINT hiprtTraversalHintDefault
#endif

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomainShadowSplit.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

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
 * Trace selection kernel of the shadow split union domain spatial reuse (see SpatialSPMISUnionDomainShadowSplit.h): the stage-2 rays of the pool
 * picks and the canonical probe rays recorded by the prepare kernel, block-distributed (spmis_union_domain_shadow_split_trace_rays_block()), one
 * visibility bit per ray. The bits of the duplicate rays are copied from the rays they repeat (spmis_union_domain_shadow_split_copy_duplicate_visibility()).
 * One thread per pixel over the 8 x 8 blocks of the render resolution.
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitTraceSelection()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_SpatialReuseSPMISUnionDomainShadowSplitTraceSelection(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_RENDER_DATA);
#endif

	// Only compiling this whole kernel if we're using stochastic pairwise MIS: the union domain reuse is not used with the other MIS weights
#if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS

	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers =
		render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers;

#ifdef __KERNELCC__
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
#endif

	// No early return before the block-distributed tracing (barriers): a thread without pixel (border blocks) or whose pixel is done has no ray
	bool thread_has_pixel			= x < render_data.render_settings.render_resolution.x && y < render_data.render_settings.render_resolution.y;
	unsigned int center_pixel_index = thread_has_pixel ? x + y * render_data.render_settings.render_resolution.x : 0;
	unsigned int prepare_flags =
		thread_has_pixel ? shadow_split_buffers.prepare_flags[center_pixel_index] : RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT;

	bool pixel_done		  = (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT) != 0;
	unsigned int ray_mask = pixel_done ? 0u : (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_SELECTION_RAY_MASK);

	// The duplicate map that the prepare kernel left in the visibility word of a pixel with duplicate rays (overwritten below). The duplicates have no
	// ray bit: only their sources are traced
	bool has_duplicates		   = !pixel_done && (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_DUPLICATES_BIT);
	unsigned int duplicate_map = has_duplicates ? shadow_split_buffers.selection_visibility[center_pixel_index] : 0u;

	unsigned int visible_mask = spmis_union_domain_shadow_split_trace_rays_block<false>(render_data, center_pixel_index, ray_mask, prepare_flags);
	if (has_duplicates)
		visible_mask = spmis_union_domain_shadow_split_copy_duplicate_visibility(visible_mask, duplicate_map);

	if (!pixel_done)
		shadow_split_buffers.selection_visibility[center_pixel_index] = visible_mask;
#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_H
