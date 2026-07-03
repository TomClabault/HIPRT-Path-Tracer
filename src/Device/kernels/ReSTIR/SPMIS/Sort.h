/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_SORT_H
#define KERNELS_RESTIR_SPMIS_SORT_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_Sort(
	unsigned int* all_pixel_hashes, unsigned int* all_pixels_index_in_cell, unsigned int* cell_offsets, unsigned int* pixel_indices_sorted, unsigned int size)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_Sort(unsigned int* all_pixel_hashes,
						 unsigned int* all_pixels_index_in_cell,
						 unsigned int* cell_offsets,
						 unsigned int* pixel_indices_sorted,
						 unsigned int size,
						 int linear_pixel_index)
#endif
{
#ifdef __KERNELCC__
	const uint32_t linear_pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if (linear_pixel_index >= size)
		return;

	unsigned int cell_index = all_pixel_hashes[linear_pixel_index];
	if (cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		return;

	unsigned int index_in_cell = all_pixels_index_in_cell[linear_pixel_index];
	unsigned int cell_offset   = cell_offsets[cell_index];

	pixel_indices_sorted[cell_offset + index_in_cell] = linear_pixel_index;
}

#endif
