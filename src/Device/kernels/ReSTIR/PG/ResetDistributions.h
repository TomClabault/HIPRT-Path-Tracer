/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_PG_RESET_DISTRIBUTIONS_H
#define DEVICE_KERNELS_RESTIR_PG_RESET_DISTRIBUTIONS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Kernel dispatched in 1D with 1 thread per cell per VMF mixture component
 */
#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PG_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PG_ResetDistributions()
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PG_ResetDistributions(HIPRTRenderData render_data, int x)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PG_RENDER_DATA);

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int cell_index		 = x / ReSTIRPGDistributionComponentCount;
	unsigned int component_index = x % ReSTIRPGDistributionComponentCount;

	if (cell_index >= render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells)
		return;

	VMF vmf;
	vmf.axis	  = fibonacci_sphere_direction(component_index, ReSTIRPGDistributionComponentCount);
	vmf.sharpness = 50.0f;
	render_data.render_settings.restir_pg_settings.hash_grid_distributions_soa.set_distribution_component_vmf(cell_index, component_index, vmf);

	float weight = 1.0f / (float)ReSTIRPGDistributionComponentCount;
	render_data.render_settings.restir_pg_settings.hash_grid_distributions_soa.set_distribution_component_weight(cell_index, component_index, weight);
}

#endif // #ifndef DEVICE_KERNELS_RESTIR_PG_RESET_DISTRIBUTIONS_H
