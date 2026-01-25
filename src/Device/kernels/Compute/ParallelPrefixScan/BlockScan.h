/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_PARALLEL_PREFIX_SCAN_BLOCK_SCAN_H
#define DEVICE_KERNELS_COMPUTE_PARALLEL_PREFIX_SCAN_BLOCK_SCAN_H

#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Device/includes/FixIntellisense.h"

 /**
  * Prefix scans the input in chunks of PARALLEL_PREFIX_SCAN_CHUNK_SIZE and outputs the block scans to output_blocks.
  * The input buffer must be padded to be multiple of PARALLEL_PREFIX_SCAN_CHUNK_SIZE elements.
  *
  * This kernel should be launched with a 1D grid of blocks with size PARALLEL_PREFIX_SCAN_CHUNK_SIZE/2 threads
  * and blocks of size PARALLEL_PREFIX_SCAN_CHUNK_SIZE/2.
  */
GLOBAL_KERNEL_SIGNATURE(void) ParallelPrefixScan_BlockScan(
	const unsigned int* const __restrict__ input,
	unsigned int* __restrict__ output_blocks,
	unsigned int* __restrict__ block_sums,
	unsigned int size)
{
	__shared__ unsigned int temp_smem[PARALLEL_PREFIX_SCAN_CHUNK_SIZE + PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2];

	unsigned int tid = threadIdx.x;
	if (tid >= PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2)
		return;

	int input_1_index = tid;
	int input_2_index = tid + (PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2);

	temp_smem[input_1_index + CONFLICT_FREE_OFFSET(input_1_index)] = input[blockIdx.x * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + input_1_index];
	temp_smem[input_2_index + CONFLICT_FREE_OFFSET(input_2_index)] = input[blockIdx.x * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + input_2_index];

	int offset = 1;
	for (int d = size >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp_smem[bi] += temp_smem[ai];
		}

		offset *= 2;
	}

	if (tid == 0)
	{
		if (block_sums != nullptr)
			// If we want the block sums
			block_sums[blockIdx.x] = temp_smem[PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1 + CONFLICT_FREE_OFFSET(PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1)];

		temp_smem[PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1 + CONFLICT_FREE_OFFSET(PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1)] = 0;
	}

	for (int d = 1; d < size; d *= 2)
	{
		__syncthreads();

		offset >>= 1;

		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int t = temp_smem[ai];
			temp_smem[ai] = temp_smem[bi];
			temp_smem[bi] += t;
		}
	}

	__syncthreads();

	if (blockIdx.x * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + 2 * tid < size)
		output_blocks[blockIdx.x * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + 2 * tid + 0] = temp_smem[2 * tid + 0 + CONFLICT_FREE_OFFSET(2 * tid + 0)];
	if (blockIdx.x * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + 2 * tid + 1 < size)
		output_blocks[blockIdx.x * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + 2 * tid + 1] = temp_smem[2 * tid + 1 + CONFLICT_FREE_OFFSET(2 * tid + 1)];
}

#endif // DEVICE_KERNELS_COMPUTE_PARALLEL_PREFIX_SCAN_BLOCK_SCAN_H
