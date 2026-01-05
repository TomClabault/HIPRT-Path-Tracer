/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Compute/RadixSortCommon.h"
#include "HostDeviceCommon/Maths/Math.h"

/**
 * The input to the count kernel is split in nChunks parts.
 * 
 * count_tables is then [nChunks * RADIX_SORT_RADIX_SIZE] in size such that each input chunk
 * can be counted separately to avoid atomic contention.
 */
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Count(
	unsigned int* __restrict__ keys,
	unsigned int* __restrict__ count_tables,
	unsigned int size,
	int bit_offset)
{
	unsigned int chunk_count = (size + RADIX_SORT_INPUT_CHUNK_SIZE - 1) / RADIX_SORT_INPUT_CHUNK_SIZE;

	for (int i = threadIdx.x; i < RADIX_SORT_INPUT_CHUNK_SIZE; i += blockDim.x)
	{
		unsigned int key_index = blockIdx.x * RADIX_SORT_INPUT_CHUNK_SIZE + i;
		if (key_index >= size)
			return;

		// Extract the radix digit (8 bits) from the key
		unsigned int key = keys[key_index];
		unsigned int radix = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

		// Atomic increment to count occurrences of each radix value
		// 
		// Also, this is storing the counts in column major order such that the prefix
		// scan that follows this kernel can be done in one go on the whole column
		// major count tables
		unsigned int count_table_index = blockIdx.x;
		hippt::atomic_fetch_add(&count_tables[count_table_index + chunk_count * radix], 1u);
	}
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H
