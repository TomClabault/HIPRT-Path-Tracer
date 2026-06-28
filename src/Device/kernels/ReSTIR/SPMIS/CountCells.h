/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_COUNT_CELLS_H
#define KERNELS_RESTIR_SPMIS_COUNT_CELLS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_CountCells(unsigned int* all_pixel_hashes,
						unsigned int* all_pixel_index_in_cell,
						unsigned short int* cell_counters,
						unsigned short int* cell_non_zero_reservoir_counters,
						unsigned int* cell_confidence_sums,
						ReSTIRPTReservoir* reservoirs,
						unsigned int size,
						bool count_important)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_CountCells(unsigned int* all_pixel_hashes,
							   unsigned int* all_pixel_index_in_cell,
							   AtomicType<unsigned short int>* cell_counters,
							   AtomicType<unsigned short int>* cell_non_zero_reservoir_counters,
							   AtomicType<unsigned int>* cell_confidence_sums,
							   ReSTIRPTReservoir* reservoirs,
							   unsigned int size,
							   bool count_important,
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

	bool important = reservoirs[linear_pixel_index].UCW > 0.0f;
	if (count_important && !important)
		return;

	if ((count_important && important) || (!count_important && !important))
	{
		unsigned short int index_in_cell = hippt::atomic_fetch_add(&cell_counters[cell_index], (unsigned short int)1);
		hippt::atomic_fetch_add(&cell_confidence_sums[cell_index], (unsigned int)reservoirs[linear_pixel_index].M);
		if (important)
			hippt::atomic_fetch_add(&cell_non_zero_reservoir_counters[cell_index], (unsigned short int)1);

		all_pixel_index_in_cell[linear_pixel_index] = index_in_cell;
	}
}

#endif
