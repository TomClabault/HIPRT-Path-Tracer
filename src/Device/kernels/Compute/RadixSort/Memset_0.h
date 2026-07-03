/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_MEMSET_0_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_MEMSET_0_H

#include "Device/includes/FixIntellisense.h"

GLOBAL_KERNEL_SIGNATURE(void)
RadixSort_Memset_0(unsigned int* __restrict__ global_count_table,
				   unsigned int global_count_table_size,
				   unsigned int* __restrict__ per_block_count_tables,
				   unsigned int per_block_count_tables_size,
				   unsigned int* __restrict__ per_block_count_tables_scanned,
				   unsigned int per_block_count_tables_scanned_size)
{
	unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_thread_index < global_count_table_size)
		global_count_table[global_thread_index] = 0;

	if (global_thread_index < per_block_count_tables_size)
		per_block_count_tables[global_thread_index] = 0;

	if (global_thread_index < per_block_count_tables_scanned_size)
		per_block_count_tables_scanned[global_thread_index] = 0;
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_MEMSET_0_H
