/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_RESET_BUFFERS_H
#define KERNELS_RESTIR_SPMIS_RESET_BUFFERS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_ResetBuffers(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_ResetBuffers(HIPRTRenderData render_data, int index)
#endif
{
#ifdef __KERNELCC__
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if (index >= render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y)
		return;

	ReSTIRCommonSPMISSettings spmis_settings = render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings;

	spmis_settings.all_pixel_hashes[index]				   = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	spmis_settings.all_pixel_hashes_checksums[index]	   = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	spmis_settings.all_pixels_index_in_cell[index]		   = 0;
	spmis_settings.important_pixel_hashes[index]		   = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	spmis_settings.pixel_indices_sorted[index]			   = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	spmis_settings.cell_pixels_counters[index]			   = 0;
	spmis_settings.cell_non_zero_reservoir_counters[index] = 0;
	spmis_settings.cell_offsets[index]					   = 0;
	spmis_settings.cell_confidence_sums[index]			   = 0;
}

#endif
