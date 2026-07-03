/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
#define DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H

#include "Device/includes/Compute/RadixSortBlock.h"
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
				  int bit_offset,
				  bool ascending_order)
{
	// Get the position for this element using atomic increment
	//
	// This gives us the exclusive prefix sum position
	constexpr int warp_size		 = 32;
	constexpr int warp_size_mask = warp_size - 1;
	constexpr int warp_count	 = RADIX_SORT_INPUT_CHUNK_SIZE / warp_size;
	unsigned int index			 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int warp_index		 = threadIdx.x / warp_size;
	unsigned int lane_index		 = threadIdx.x & warp_size_mask;

	unsigned int key   = index < size ? (ascending_order ? input_keys[index] : ~input_keys[index]) : 0;
	unsigned int value = index < size ? input_values[index] : 0;
	unsigned int radix = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

	unsigned int num_blocks	   = (size + RADIX_SORT_INPUT_CHUNK_SIZE - 1) / RADIX_SORT_INPUT_CHUNK_SIZE;
	unsigned int global_offset = index < size ? global_count_table_prefix_scanned[radix] : 0;
	unsigned int block_offset  = index < size ? per_block_count_table_prefix_scanned[radix * num_blocks + blockIdx.x] : 0;

	__shared__ short int digit_count_per_warp[warp_count][RADIX_SORT_RADIX_SIZE];

	for (int i = lane_index; i < RADIX_SORT_RADIX_SIZE; i += warp_size)
		digit_count_per_warp[warp_index][i] = 0;
	__syncthreads();

	if (index < size)
		hippt::atomic_fetch_add_gpu(&digit_count_per_warp[warp_index][radix], (short int)1);

	__syncthreads(); // We now have the count of each radix digit for each warp, we can compute inter-warp offset by summing the counts of the previous warps
					 // (prefix scan)

	__shared__ short int digit_count_per_warp_prefix_summed[warp_count][RADIX_SORT_RADIX_SIZE];
	for (int digit = warp_index; digit < RADIX_SORT_RADIX_SIZE; digit += warp_count)
	{
		int my_digit_count			= digit_count_per_warp[lane_index][digit];
		int my_digit_prefix_scanned = warp_prefix_scan_exclusive(my_digit_count);

		digit_count_per_warp_prefix_summed[lane_index][digit] = my_digit_prefix_scanned;
	}

	__syncthreads();

	unsigned int inter_warp_offset = digit_count_per_warp_prefix_summed[warp_index][radix];

	// And now computing the intra-warp offset by summing the counts of the previous threads with the same radix in the same warp
	unsigned int intra_warp_offset = 0;
	for (int l = 0; l < warp_size; l++)
	{
		// For each lane, ask the other threads in the warp (ballot) if they have the same value and count how many of them are before us (mask_lane_lt + popc)
		unsigned int radix_lane		 = hippt::warp_shfl(radix, l);
		unsigned int same_radix_mask = hippt::warp_ballot(0xFFFFFFFF, radix_lane == radix);
		unsigned int mask_lane_lt	 = (1U << lane_index) - 1;
		if (lane_index == l)
			// For this thread, the intra-warp offset is the number of threads with the same radix that are before us in the warp, which is given by the
			// population count of the mask of threads with the same radix that are before us
			intra_warp_offset = hippt::popc(same_radix_mask & mask_lane_lt);
	}

	unsigned int sorted_position = global_offset + block_offset + inter_warp_offset + intra_warp_offset;

	// TODO do sorting in shared memory before scattering to global memory such that writes are coalesced to global memory
	// Scatter to the output position
	if (index < size)
	{
		output_keys[sorted_position]   = ascending_order ? key : ~key;
		output_values[sorted_position] = value;
	}
}

#endif // DEVICE_KERNELS_COMPUTE_RADIX_SORT_REORDER_H
