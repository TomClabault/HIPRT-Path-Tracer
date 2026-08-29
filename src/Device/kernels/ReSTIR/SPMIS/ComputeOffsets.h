/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_COMPUTE_OFFSETS_H
#define KERNELS_RESTIR_SPMIS_COMPUTE_OFFSETS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_ComputeOffsets(unsigned short int* cell_counters, unsigned int* global_cell_offset_counter, unsigned int* cell_offsets, unsigned int size)
#else // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_ComputeOffsets(AtomicType<unsigned short int>* cell_counters,
								   AtomicType<unsigned int>* global_cell_offset_counter,
								   unsigned int* cell_offsets,
								   unsigned int size,
								   unsigned int x)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__

	if (x >= size)
		return;

	unsigned int hash_cell_index  = x;
	unsigned short int cell_count = hippt::atomic_load(&cell_counters[hash_cell_index]);
	unsigned int cell_offset	  = hippt::atomic_fetch_add(global_cell_offset_counter, (unsigned int)cell_count);

	cell_offsets[hash_cell_index] = cell_offset;
}

#endif // #ifndef KERNELS_RESTIR_SPMIS_COMPUTE_OFFSETS_H
