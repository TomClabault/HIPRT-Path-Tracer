/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_RADIX_SORT_BLOCK_H
#define DEVICE_INCLUDES_COMPUTE_RADIX_SORT_BLOCK_H

#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

#define RadixSortBlockRadixBits 4
#define RadixSortBlockRadix		(1 << RadixSortBlockRadixBits)

/**
 * keys and values are assumed to be array_length in size
 * and array_length is a multiple of 32
 */
template <int array_length, int key_bits, typename K_t, typename V_t>
HIPRT_DEVICE void radix_threadblock_sort_key_values(K_t* keys, V_t* values)
{
	__shared__ int digit_counts[RadixSortBlockRadix];
	__shared__ K_t temp_keys[array_length];
	__shared__ V_t temp_values[array_length];

	K_t* in_keys	= keys;
	V_t* in_values	= values;
	K_t* out_keys	= temp_keys;
	V_t* out_values = temp_values;

	const int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;
	const int threads_per_block	 = blockDim.x * blockDim.y;
	const int warp_index		 = thread_id_in_block >> 5;
	const int lane_idx			 = thread_id_in_block & 31;
	constexpr int warp_count	 = array_length >> 5;

	// Assert array length divisible by warp size
	static_assert(warp_count << 5 == array_length);

	// Key bits divisible by radix bit
	static_assert(key_bits / RadixSortBlockRadixBits * RadixSortBlockRadixBits == key_bits);

	__syncthreads();

	for (int shift = 0; shift < key_bits; shift += RadixSortBlockRadixBits)
	{
		// Reset digit counts
		if (thread_id_in_block < RadixSortBlockRadix)
			digit_counts[thread_id_in_block] = 0;
		__syncthreads();

		// Count digits
		int digit = (in_keys[thread_id_in_block] >> shift) & (RadixSortBlockRadix - 1);
		hippt::atomic_fetch_add_gpu(&digit_counts[digit], 1);

		__syncthreads();

		// Compute prefix sums for each digit bin
		__shared__ int bucket_base[RadixSortBlockRadix];
		if constexpr (RadixSortBlockRadix < 32)
		{
			if (warp_index == 0)
			{
				int bucket_base_value = warp_prefix_scan_exclusive(thread_id_in_block < RadixSortBlockRadix ? digit_counts[thread_id_in_block] : 0,
																   thread_id_in_block);

				if (thread_id_in_block < RadixSortBlockRadix)
					bucket_base[thread_id_in_block] = bucket_base_value;
			}
		}
		else if (threads_per_block >= RadixSortBlockRadix)
		{
			int digit_count = thread_id_in_block < RadixSortBlockRadix ? digit_counts[thread_id_in_block] : 0;
			int digit_scan	= block_prefix_scan_exclusive<RadixSortBlockRadix>(digit_count, thread_id_in_block);
			if (thread_id_in_block < RadixSortBlockRadix)
				bucket_base[thread_id_in_block] = digit_scan;
		}
		else
		{
			// Simple fallback, not performant
			if (thread_id_in_block == 0)
			{
				int sum = 0;
				for (int i = 0; i < RadixSortBlockRadix; i++)
				{
					bucket_base[i] = sum;
					sum += digit_counts[i];
				}
			}
		}

		__syncthreads();

		// Computing the local rank of each element within its digit bin
		unsigned int warp_local_rank = 0;
		__shared__ unsigned int warp_digit_counts[warp_count][RadixSortBlockRadix];
		for (int d = 0; d < RadixSortBlockRadix; d++)
		{
			bool is_digit = digit == d;

			unsigned int ballot		  = hippt::warp_ballot(0xffffffff, is_digit);
			unsigned int mask_lane_lt = (lane_idx == 0) ? 0u : (1u << lane_idx) - 1;

			if (is_digit)
				warp_local_rank = hippt::popc(ballot & mask_lane_lt);

			if (lane_idx == 0)
				warp_digit_counts[warp_index][d] = hippt::popc(ballot);
		}

		__syncthreads();

		__shared__ unsigned int warp_digit_bases[warp_count][RadixSortBlockRadix];
		// TODO We can have each warp scan one digit here
		if (warp_index == 0)
		{
			for (int d = 0; d < RadixSortBlockRadix; d++)
			{
				unsigned int warp_digit_count  = thread_id_in_block < warp_count ? warp_digit_counts[thread_id_in_block][d] : 0;
				unsigned int warp_digit_prefix = warp_prefix_scan_exclusive(warp_digit_count, thread_id_in_block);

				if (thread_id_in_block < warp_count)
					warp_digit_bases[thread_id_in_block][d] = warp_digit_prefix;
			}
		}

		__syncthreads();

		// Scatter step
		unsigned int warp_digit_base = warp_digit_bases[warp_index][digit];
		int output_index			 = bucket_base[digit] + warp_local_rank + warp_digit_base;
		out_keys[output_index]		 = in_keys[thread_id_in_block];
		out_values[output_index]	 = in_values[thread_id_in_block];

		__syncthreads();

		// Swap in_keys and out_keys for the next iteration
		K_t* temp = in_keys;
		in_keys	  = out_keys;
		out_keys  = temp;

		V_t* temp_v = in_values;
		in_values	= out_values;
		out_values	= temp_v;
	}

	if (key_bits / RadixSortBlockRadixBits & 1)
	{
		// Copying so the final output is in the input arrays given by the user
		in_keys[thread_id_in_block]	  = out_keys[thread_id_in_block];
		in_values[thread_id_in_block] = out_values[thread_id_in_block];
	}
}

#endif
