/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H

#include "Device/includes/FixIntellisense.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Reorder(
	unsigned int* __restrict__ input_keys,
	unsigned int* __restrict__ input_values,
	unsigned int* __restrict__ output_keys,
	unsigned int* __restrict__ output_values,
	unsigned int* __restrict__ count_table,
	unsigned int size,
	int bit_offset)
#else
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Reorder(
	unsigned int* __restrict__ input_keys,
	unsigned int* __restrict__ input_values,
	unsigned int* __restrict__ output_keys,
	unsigned int* __restrict__ output_values,
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

	// Extract the radix digit from the key
	constexpr unsigned int RADIX_MASK = 0xFF; // 8 bits
	unsigned int key = input_keys[thread_index];
	unsigned int value = input_values[thread_index];
	unsigned int radix = (key >> bit_offset) & RADIX_MASK;

	// Get the position for this element using atomic increment
	// This gives us the exclusive prefix sum position
#ifdef __KERNELCC__
	unsigned int position = atomicAdd(&count_table[radix], 1);
#else
	// CPU fallback (not thread-safe, but for intellisense)
	unsigned int position = count_table[radix];
	count_table[radix]++;
#endif

	// Scatter to the output position
	output_keys[position] = key;
	output_values[position] = value;
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
