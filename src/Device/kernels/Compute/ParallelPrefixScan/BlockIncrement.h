/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_PARALLEL_PREFIX_SCAN_BLOCK_INCREMENT_H
#define DEVICE_KERNELS_COMPUTE_PARALLEL_PREFIX_SCAN_BLOCK_INCREMENT_H

#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Device/includes/FixIntellisense.h"

/**
 * Adds each element of block_sums to each corresponding chunk of input ("chunks" are of size PARALLEL_PREFIX_SCAN_CHUNK_SIZE)
 *
 * This kernel should be launched with 1 thread per element of the input
 */
GLOBAL_KERNEL_SIGNATURE(void)
ParallelPrefixScan_BlockIncrement(unsigned int* __restrict__ input, const unsigned int* const __restrict__ block_sums, unsigned int size)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size)
		return;

	input[tid] += block_sums[blockIdx.x];
}

#endif
