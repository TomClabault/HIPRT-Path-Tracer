/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H

#include "Device/includes/Compute/RadixSortCommon.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Reorder(
	unsigned int* __restrict__ input_keys,
	unsigned int* __restrict__ input_values,
	unsigned int* __restrict__ output_keys,
	unsigned int* __restrict__ output_values,
	unsigned int* __restrict__ count_table,
	unsigned int size,
	int bit_offset)
{
	const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_index >= size)
		return;

	// Extract the radix digit from the key
	unsigned int key = input_keys[thread_index];
	unsigned int value = input_values[thread_index];
	unsigned int radix = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

	// Get the position for this element using atomic increment
	// This gives us the exclusive prefix sum position
	unsigned int position = hippt::atomic_fetch_add(&count_table[radix], 1u);

	// Scatter to the output position
	output_keys[position] = key;
	output_values[position] = value;
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
