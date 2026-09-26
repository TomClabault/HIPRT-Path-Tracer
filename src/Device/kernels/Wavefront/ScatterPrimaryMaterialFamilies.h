/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_SCATTER_PRIMARY_MATERIAL_FAMILIES_H
#define KERNELS_WAVEFRONT_SCATTER_PRIMARY_MATERIAL_FAMILIES_H

#include "Device/includes/Wavefront/WavefrontShading.h"

HIPRT_DEVICE static void wavefront_scatter_primary_material_family(HIPRTRenderData& render_data, unsigned int pixel_index)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	unsigned int material_family		= wavefront_data.path_material_family_tags[pixel_index];
	if (material_family >= KernelMaterialSpecializationCount)
		return;

	unsigned int destination_index = hippt::atomic_fetch_add(wavefront_data.material_family_cursors + material_family, 1u);
	if (destination_index < wavefront_data.path_capacity)
		wavefront_data.material_family_indices[destination_index] = pixel_index;
}

#ifdef __KERNELCC__
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_SCATTER_PRIMARY_MATERIAL_FAMILIES_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ScatterPrimaryMaterialFamilies()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ScatterPrimaryMaterialFamilies(HIPRTRenderData render_data, unsigned int pixel_index)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_SCATTER_PRIMARY_MATERIAL_FAMILIES_RENDER_DATA);
	unsigned int pixel_index	 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pixel_stride	 = gridDim.x * blockDim.x;
#else  // #ifdef __KERNELCC__
	unsigned int pixel_stride = render_data.wavefront_data.path_capacity;
#endif // #ifdef __KERNELCC__
	if (render_data.wavefront_data.material_family_routing_enabled == 0)
		return;

	for (; pixel_index < render_data.wavefront_data.path_capacity; pixel_index += pixel_stride)
		wavefront_scatter_primary_material_family(render_data, pixel_index);
}

#endif // #ifndef KERNELS_WAVEFRONT_SCATTER_PRIMARY_MATERIAL_FAMILIES_H
