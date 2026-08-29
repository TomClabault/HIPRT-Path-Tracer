/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_RESET_CELLS_DATA_H
#define KERNELS_RESTIR_SPMIS_RESET_CELLS_DATA_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_SPMIS_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_ResetCellsData(unsigned int size)
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_ResetCellsData(HIPRTRenderData render_data, unsigned int size, int cell_index)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_SPMIS_RENDER_DATA);

	const uint32_t cell_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__

	if (cell_index >= size)
		return;

	ReSTIRPTSPMISSettings spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

	if (cell_index == 0)
		*spmis_settings.cell_global_offset_counter = 0;

	spmis_settings.all_pixels_index_in_cell[cell_index]			= 0;
	spmis_settings.pixel_indices_sorted[cell_index]				= HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	spmis_settings.cell_pixels_counters[cell_index]				= 0;
	spmis_settings.cell_non_zero_reservoir_counters[cell_index] = 0;
	spmis_settings.cell_offsets[cell_index]						= 0;
	spmis_settings.cell_confidence_sums[cell_index]				= 0;
}

#endif // #ifndef KERNELS_RESTIR_SPMIS_RESET_CELLS_DATA_H
