/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H

#include "Device/includes/Compute/RadixSortCommon.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

/**
 * The input is split into chunks of size RADIX_SORT_INPUT_CHUNK_SIZE.
 *
 * Each block processes one chunk and produces a per-block count table of size RADIX_SORT_RADIX_SIZE. This makes the 'per_block_count_tables' buffer of size:
 * size / RADIX_SORT_INPUT_CHUNK_SIZE * RADIX_SORT_RADIX_SIZE
 */
GLOBAL_KERNEL_SIGNATURE(void)
RadixSort_Count(unsigned int* __restrict__ keys,
				unsigned int* __restrict__ count_tables,
				unsigned int* __restrict__ per_block_count_tables,
				unsigned int size,
				int bit_offset)
{
	__shared__ unsigned int per_block_histogram[RADIX_SORT_RADIX_SIZE];
	if (threadIdx.x < RADIX_SORT_RADIX_SIZE)
		per_block_histogram[threadIdx.x] = 0;
	__syncthreads();

	unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_thread_index < size)
	{
		// Extract the radix digit (8 bits) from the key
		unsigned int key   = keys[global_thread_index];
		unsigned int radix = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

		// Atomic increment to count occurrences of each radix value
		hippt::atomic_fetch_add(&count_tables[radix], 1u);
		hippt::atomic_fetch_add(&per_block_histogram[radix], 1u);
	}

	__syncthreads();

	// After this syncthreads, we have the per-block histogram in shared memory, we can now write the histogram to global memory in column major order to
	// prepare for the prefix scan step that follows.
	if (threadIdx.x < RADIX_SORT_RADIX_SIZE)
	{
		// Write back the scanned value to the per-block count table
		unsigned int radix						  = threadIdx.x;
		unsigned int count_table_index			  = blockIdx.x * RADIX_SORT_RADIX_SIZE + radix;
		per_block_count_tables[count_table_index] = per_block_histogram[radix];
	}
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H
