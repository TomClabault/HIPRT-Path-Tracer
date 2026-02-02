/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H

#include "Device/includes/Compute/RadixSortCommon.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

GLOBAL_KERNEL_SIGNATURE(void)
RadixSort_Reorder(const unsigned int* __restrict__ input_keys,
				  const unsigned int* __restrict__ input_values,
				  unsigned int* __restrict__ output_keys,
				  unsigned int* __restrict__ output_values,
				  const unsigned int* __restrict__ global_count_table_prefix_scanned,
				  const unsigned int* __restrict__ per_block_count_table_prefix_scanned,
				  unsigned int size,
				  int bit_offset)
{
	// const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	// if (thread_index >= size)
	//	return;

	//// Extract the radix digit from the key
	// unsigned int key   = input_keys[thread_index];
	// unsigned int value = input_values[thread_index];
	// unsigned int radix = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

	// Get the position for this element using atomic increment
	// This gives us the exclusive prefix sum position

	__shared__ unsigned int digit_count_in_block[RADIX_SORT_RADIX_SIZE];
	if (threadIdx.x < RADIX_SORT_RADIX_SIZE)
		digit_count_in_block[threadIdx.x] = 0;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 0; i < blockDim.x; i++)
		{
			unsigned int index = blockIdx.x * blockDim.x + i;
			if (index >= size)
				break;

			unsigned int key   = input_keys[index];
			unsigned int value = input_values[index];
			unsigned int radix = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

			unsigned int global_offset = global_count_table_prefix_scanned[radix];
			unsigned int block_offset  = per_block_count_table_prefix_scanned[blockIdx.x * RADIX_SORT_RADIX_SIZE + radix];
			unsigned int local_offset  = hippt::atomic_fetch_add(&digit_count_in_block[radix], 1u);

			unsigned int position = global_offset + block_offset + local_offset;

			// Scatter to the output position
			output_keys[position]	= key;
			output_values[position] = value;
		}
	}
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
