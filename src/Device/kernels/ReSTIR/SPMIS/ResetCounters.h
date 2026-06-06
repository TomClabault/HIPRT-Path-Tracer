/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_RESET_COUNTERS_H
#define KERNELS_RESTIR_SPMIS_RESET_COUNTERS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_ResetCounters(unsigned int* cell_pixels_counters,
						   unsigned int* cell_non_zero_reservoir_counters,
						   unsigned int* cell_confidence_sums,
						   unsigned int* cell_global_offset_counter,
						   unsigned int size)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_ResetCounters(AtomicType<unsigned int>* cell_pixels_counters,
								  AtomicType<unsigned int>* cell_non_zero_reservoir_counters,
								  AtomicType<unsigned int>* cell_confidence_sums,
								  AtomicType<unsigned int>* cell_global_offset_counter,
								  unsigned int size,
								  int cell_index)
#endif
{
#ifdef __KERNELCC__
	const uint32_t cell_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if (cell_index >= size)
		return;

	if (cell_index == 0)
		*cell_global_offset_counter = 0;

	cell_pixels_counters[cell_index]			 = 0;
	cell_non_zero_reservoir_counters[cell_index] = 0;
	cell_confidence_sums[cell_index]			 = 0;
}

#endif
