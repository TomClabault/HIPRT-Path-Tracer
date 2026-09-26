/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_RESET_MATERIAL_FAMILY_ROUTING_H
#define KERNELS_WAVEFRONT_RESET_MATERIAL_FAMILY_ROUTING_H

#include "Device/includes/Wavefront/WavefrontShading.h"

#ifdef __KERNELCC__
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_RESET_MATERIAL_FAMILY_ROUTING_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ResetMaterialFamilyRouting()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ResetMaterialFamilyRouting(HIPRTRenderData render_data)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_RESET_MATERIAL_FAMILY_ROUTING_RENDER_DATA);
	if (blockIdx.x != 0 || threadIdx.x != 0)
		return;
#endif // #ifdef __KERNELCC__
	if (render_data.wavefront_data.material_family_routing_enabled == 0)
		return;

	for (unsigned int family_index = 0; family_index < KernelMaterialSpecializationCount; family_index++)
	{
		render_data.wavefront_data.material_family_counts[family_index]	 = 0;
		render_data.wavefront_data.material_family_cursors[family_index] = 0;
	}
}

#endif // #ifndef KERNELS_WAVEFRONT_RESET_MATERIAL_FAMILY_ROUTING_H
