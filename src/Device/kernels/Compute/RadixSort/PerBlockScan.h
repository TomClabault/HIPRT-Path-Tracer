/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_PER_BLOCK_SCAN_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_PER_BLOCK_SCAN_H

#include "Device/includes/Compute/RadixSortCommon.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

// Register-based warp inclusive scan
HIPRT_DEVICE unsigned int warp_scan_inclusive(unsigned int val)
{
	unsigned int lane = threadIdx.x % 32;

	unsigned int temp = hippt::warp_shfl_up(val, 1);
	if (lane >= 1)
		val += temp;
	temp = hippt::warp_shfl_up(val, 2);
	if (lane >= 2)
		val += temp;
	temp = hippt::warp_shfl_up(val, 4);
	if (lane >= 4)
		val += temp;
	temp = hippt::warp_shfl_up(val, 8);
	if (lane >= 8)
		val += temp;
	temp = hippt::warp_shfl_up(val, 16);
	if (lane >= 16)
		val += temp;

	return val;
}

HIPRT_DEVICE unsigned int block_exclusive_scan(unsigned int thread_input_value)
{
	unsigned int tid = threadIdx.x;

	// Per-warp scan
	unsigned int warp_prefix = warp_scan_inclusive(thread_input_value);

	// The last lane of each warp holds the sum for that warp
	unsigned int lane	 = tid % 32;
	unsigned int warp_id = tid / 32;

	// Shared memory to hold the sum of each warp
	// (Size = Max Threads / 32). Assuming max 1024 threads -> 32 warps.
	constexpr unsigned int WARP_COUNT = RADIX_SORT_INPUT_CHUNK_SIZE / 32;
	__shared__ unsigned int smem_warp_sums[WARP_COUNT];

	if (lane == 31)
		// Storing the total sum of each warp in shared memory
		smem_warp_sums[warp_id] = warp_prefix;

	__syncthreads();

	// Scan the warp sums (only warp 0 does this)
	// This calculates the base value to add to each warp
	unsigned int warp_base = 0;
	if (warp_id == 0)
	{
		unsigned int my_warp_sum = 0;

		if (tid < WARP_COUNT)
			// Only load if the warp actually exists in this block
			my_warp_sum = smem_warp_sums[tid];

		unsigned int inclusive_warp_sum_scan = warp_scan_inclusive(my_warp_sum);

		// Write the inclusive scan back to smem so other warps can read their "base"
		// Note: We shift by 1 index effectively, because Warp N needs the sum of Warps 0..N-1
		// We can store it directly and handle the shift on read.
		smem_warp_sums[tid] = inclusive_warp_sum_scan;
	}

	__syncthreads();

	// Add the base from previous warps to the local warp prefix
	if (warp_id > 0)
		warp_base = smem_warp_sums[warp_id - 1];

	// This thread's inclusive sum over the whole block
	//
	// To convert to exclusive, we subtract the thread's input value
	return warp_prefix + warp_base - thread_input_value;
}

/**
 * This kernel expects workgroups of size RADIX_SORT_RADIX_SIZE
 */
GLOBAL_KERNEL_SIGNATURE(void) RadixSort_PerBlockScan(unsigned int* __restrict__ in_out_per_block_count_tables, unsigned int size)
{
	unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_thread_index >= size)
		return;

	unsigned int value		   = in_out_per_block_count_tables[global_thread_index];
	unsigned int scanned_value = block_exclusive_scan(value);

	// Store back the scanned value to global memory
	in_out_per_block_count_tables[global_thread_index] = scanned_value;
}

#endif