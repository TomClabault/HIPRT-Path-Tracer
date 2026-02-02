/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_PER_BLOCK_SCAN_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_PER_BLOCK_SCAN_H

#include "Device/includes/Compute/RadixSortCommon.h"
#include "Device/includes/FixIntellisense.h"

/**
 * This kernel expects 1 workgroup of size RADIX_SORT_RADIX_SIZE
 *
 * The output is formatted as: [ block0_radix0, block0_radix1, ..., block0_radix255, block1_radix0, block1_radix1, ..., block1_radix255, ... ]
 */
GLOBAL_KERNEL_SIGNATURE(void)
RadixSort_PerBlockScan(const unsigned int* __restrict__ in_per_block_count_tables,
					   unsigned int* __restrict__ out_scanned_per_block_count_tables,
					   unsigned int num_blocks)
{
	if (blockIdx.x >= num_blocks)
		return;

	const unsigned int digit = threadIdx.x;

	// Each threads does a sequential scan to compute the exclusive prefix sum for its radix accross all block-count-tables
	unsigned int exclusive_prefix_sum = 0;
	for (int i = 0; i < num_blocks; i++)
	{
		unsigned int index		 = i * RADIX_SORT_RADIX_SIZE + digit;
		unsigned int input_value = in_per_block_count_tables[index];

		out_scanned_per_block_count_tables[index] = exclusive_prefix_sum;

		exclusive_prefix_sum += input_value;
	}
}

#endif
