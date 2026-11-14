/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_H

// The shadow rays of this kernel are the mostly visible rays of the single union domain kernel (SpatialReuseSPMISUnionDomain.h): they keep the
// default traversal hint as there (Intersect.h). Defined before the includes so that Intersect.h sees it. GPU only: the CPU renderer compiles every
// kernel in one translation unit
#ifdef __KERNELCC__
#define SHADOW_RAY_TRAVERSAL_HINT hiprtTraversalHintDefault
#endif

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomainShadowSplit.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Trace selection kernel of the shadow split union domain spatial reuse (see SpatialSPMISUnionDomainShadowSplit.h): the stage-2 rays of the pool
 * picks and the canonical probe rays recorded by the prepare kernel, block-distributed (spmis_union_domain_shadow_split_trace_rays_block()), one
 * visibility bit per ray. One thread per pixel over the 8 x 8 blocks of the render resolution, as the single union domain kernel. GPU only.
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
	// GPU only (shared memory and barriers of the block-distributed tracing): the CPU renderer runs the single union domain kernel
#if defined(__KERNELCC__) && ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_RENDER_DATA);

	const ReSTIRPTSPMISUnionDomainShadowSplitBuffers& shadow_split_buffers =
		render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.shadow_split_buffers;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// No early return before the block-distributed tracing (barriers): a thread without pixel (border blocks) or whose pixel is done has no ray
	bool thread_has_pixel			= x < render_data.render_settings.render_resolution.x && y < render_data.render_settings.render_resolution.y;
	unsigned int center_pixel_index = thread_has_pixel ? x + y * render_data.render_settings.render_resolution.x : 0;
	unsigned int prepare_flags =
		thread_has_pixel ? shadow_split_buffers.prepare_flags[center_pixel_index] : RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT;

	bool pixel_done		  = (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_PIXEL_DONE_BIT) != 0;
	unsigned int ray_mask = pixel_done ? 0u : (prepare_flags & RESTIR_PT_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_SELECTION_RAY_MASK);

	unsigned int visible_mask = spmis_union_domain_shadow_split_trace_rays_block<false>(render_data, center_pixel_index, ray_mask, prepare_flags);

	if (!pixel_done)
		shadow_split_buffers.selection_visibility[center_pixel_index] = visible_mask;
#endif // #if defined(__KERNELCC__) && ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS
}

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_TRACE_SELECTION_H
