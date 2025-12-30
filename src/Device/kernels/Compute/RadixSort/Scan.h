/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_SCAN_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_SCAN_H

#include "Device/includes/FixIntellisense.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Scan(
	unsigned int* __restrict__ count_table)
#else
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_Scan(
	unsigned int* __restrict__ count_table,
	int thread_index)
#endif
{
	constexpr int RADIX_SIZE = 256; // 2^8

#ifdef __KERNELCC__
	const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	// Simple sequential scan (exclusive prefix sum)
	// For 256 elements, a single thread doing sequential scan is efficient
	// and avoids the complexity of a parallel scan
	if (thread_index == 0)
	{
		unsigned int sum = 0;
		for (int i = 0; i < RADIX_SIZE; i++)
		{
			unsigned int count = count_table[i];
			count_table[i] = sum;
			sum += count;
		}
	}
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_SCAN_H
