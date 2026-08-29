/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_PG_RESET_HASH_GRID_H
#define DEVICE_KERNELS_RESTIR_PG_RESET_HASH_GRID_H

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
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PG_ResetHashGrid()
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PG_ResetHashGrid(HIPRTRenderData render_data, int x)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PG_RENDER_DATA);

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int cell_index = x;
	if (cell_index >= render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells)
		return;

	if (cell_index == 0)
		*render_data.render_settings.restir_pg_settings.grid_cell_alive_count = 0;

	render_data.render_settings.restir_pg_settings.hash_grid_checksums[cell_index] = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	const ReSTIRPGDistributionSufficientStatisticsSoADevice& soa_device =
		render_data.render_settings.restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa;
	for (unsigned int i = 0; i < ReSTIRPGDistributionComponentCount; i++)
	{
		unsigned int index = i * render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells + cell_index;

		soa_device.directions_sum_x[index]			 = 0.0f;
		soa_device.directions_sum_y[index]			 = 0.0f;
		soa_device.directions_sum_z[index]			 = 0.0f;
		soa_device.responsibility_weights_sum[index] = 0.0f;
	}

	soa_device.sample_count[cell_index] = 0u;
}

#endif // #ifndef DEVICE_KERNELS_RESTIR_PG_RESET_HASH_GRID_H
