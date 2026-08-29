/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_PG_RESET_SUFFICIENT_STATISTICS_H
#define DEVICE_KERNELS_RESTIR_PG_RESET_SUFFICIENT_STATISTICS_H

#include "Device/includes/FixIntellisense.h"
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
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PG_ResetSufficientStatistics()
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PG_ResetSufficientStatistics(HIPRTRenderData render_data, int x)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PG_RENDER_DATA);

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__

	ReSTIRPGSettings& restir_pg_settings = render_data.render_settings.restir_pg_settings;

	unsigned int cell_index		 = x / ReSTIRPGDistributionComponentCount;
	unsigned int component_index = x % ReSTIRPGDistributionComponentCount;

	if (cell_index >= restir_pg_settings.hash_grid_total_number_of_cells)
		return;

	ReSTIRPGDistributionSufficientStatisticsSoADevice& sufficient_statistics = restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa;
	// if (hash_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		// We reset the sufficient statistics of the distribution of this cell and this component to 0. This is necessary when we reset the render because we
		// want to start accumulating new sufficient statistics for the new render and not keep the old ones that were accumulated for the previous render.

		unsigned int index										= component_index * restir_pg_settings.hash_grid_total_number_of_cells + cell_index;
		sufficient_statistics.directions_sum_x[index]			= 0.0f;
		sufficient_statistics.directions_sum_y[index]			= 0.0f;
		sufficient_statistics.directions_sum_z[index]			= 0.0f;
		sufficient_statistics.responsibility_weights_sum[index] = 0.0f;

		sufficient_statistics.sample_count[cell_index] = 0;
	}
}

#endif // #ifndef DEVICE_KERNELS_RESTIR_PG_RESET_SUFFICIENT_STATISTICS_H
