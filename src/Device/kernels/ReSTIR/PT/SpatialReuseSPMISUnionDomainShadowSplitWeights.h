/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_H
#define DEVICE_RESTIR_PT_SPATIAL_REUSE_SPMIS_UNION_DOMAIN_SHADOW_SPLIT_WEIGHTS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PT/SpatialSPMISUnionDomainShadowSplit.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"

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
