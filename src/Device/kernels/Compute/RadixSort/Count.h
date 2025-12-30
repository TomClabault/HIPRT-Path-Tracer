/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H

#include "Device/includes/FixIntellisense.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Count(
	unsigned int* __restrict__ keys,
	unsigned int* __restrict__ count_table,
	unsigned int size,
	int bit_offset)
#else
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Count(
	unsigned int* __restrict__ keys,
	unsigned int* __restrict__ count_table,
	unsigned int size,
	int bit_offset,
	int thread_index)
#endif
{
#ifdef __KERNELCC__
	const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if (thread_index >= size)
		return;

	// Extract the radix digit (8 bits) from the key
	constexpr unsigned int RADIX_MASK = 0xFF; // 8 bits
	unsigned int key = keys[thread_index];
	unsigned int radix = (key >> bit_offset) & RADIX_MASK;

	// Atomic increment to count occurrences of each radix value
#ifdef __KERNELCC__
	atomicAdd(&count_table[radix], 1);
#else
	// CPU fallback (not thread-safe, but for intellisense)
	count_table[radix]++;
#endif
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_COUNT_H
