/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_RADIX_SORT_BLOCK_H
#define DEVICE_INCLUDES_COMPUTE_RADIX_SORT_BLOCK_H

#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

#define RadixSortBlockRadixBits 2
#define RadixSortBlockRadix		(1 << RadixSortBlockRadixBits)

#define DEBUG_PRINT 0

/**
 * keys and values are assumed to be array_length in size
 * and array_length is a multiple of 32
 */
template <int array_length, typename K_t, typename V_t>
HIPRT_DEVICE void radix_threadblock_sort(K_t* keys, V_t* values)
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
	const int warp_index		 = thread_id_in_block / 32;

	bool DEBUGvery_first = thread_id_in_block == 0 && blockIdx.x == 0 && blockIdx.y == 0;
	if (DEBUGvery_first)
	{
#if DEBUG_PRINT
		printf("Input values:\n");
		printf("\t[");
		for (int i = 0; i < array_length; i++)
			printf("(%d, %d), ", in_keys[i], in_values[i]);
		printf("\n");
		printf("\n");
#endif
	}

	__syncthreads();

	for (int shift = 0; shift < sizeof(K_t) * 8; shift += RadixSortBlockRadixBits)
	{
		if (DEBUGvery_first)
		{
#if DEBUG_PRINT
			printf("Pass %d\n", shift);
#endif
		}

		// Reset digit counts
		if (thread_id_in_block < RadixSortBlockRadix)
			digit_counts[thread_id_in_block] = 0;
		__syncthreads();

		// Count digits
		int digit = (in_keys[thread_id_in_block] >> shift) & (RadixSortBlockRadix - 1);
#if defined(__KERNELCC__) // Needs this otherwise the CPU compiler whines because hippt::atomic_fetch_add is called on a non-atomic variable. On the GPU this is
						  // fine because
		hippt::atomic_fetch_add(&digit_counts[digit], 1);
#endif
		if (DEBUGvery_first && DEBUG_PRINT)
		{
#if DEBUG_PRINT
			printf("Digit counts:\n");
			printf("\t[");
			for (int i = 0; i < RadixSortBlockRadix; i++)
				printf("%2d, ", digit_counts[i]);
			printf("\n");
#endif
		}

		__syncthreads();

		// Compute prefix sums for each digit bin
		__shared__ int bucket_base[RadixSortBlockRadix];
		if (warp_index == 0)
		{
			int bucket_base_value = warp_scan_exclusive(thread_id_in_block < RadixSortBlockRadix ? digit_counts[thread_id_in_block] : 0, thread_id_in_block);
			if (thread_id_in_block < RadixSortBlockRadix)
				bucket_base[thread_id_in_block] = bucket_base_value;
		}

		__syncthreads();

		if (DEBUGvery_first && DEBUG_PRINT)
		{
#if DEBUG_PRINT
			printf("Bucket bases:\n");
			printf("\t[");
			for (int i = 0; i < RadixSortBlockRadix; i++)
				printf("%d, ", bucket_base[i]);
			printf("\n");
#endif
		}

		// Computing the local rank of each element within its digit bin
		int local_rank = 0;
		for (int i = 0; i < RadixSortBlockRadix; i++)
		{
			bool is_digit = digit == i;
			int rank	  = block_scan_exclusive<array_length>(is_digit, thread_id_in_block);
			__syncthreads();

			if (is_digit)
				local_rank = rank;
		}

		{
			__shared__ int local_rank_shared[array_length];
			local_rank_shared[thread_id_in_block] = local_rank;
			__syncthreads();

			if (DEBUGvery_first && DEBUG_PRINT)
			{
#if DEBUG_PRINT
				printf("Local ranks:\n");
				printf("\t[");
				for (int i = 0; i < array_length; i++)
					printf("%2d, ", local_rank_shared[i]);
				printf("\n");
#endif
			}
		}

		// Scatter step
		int output_index		 = bucket_base[digit] + local_rank;
		out_keys[output_index]	 = in_keys[thread_id_in_block];
		out_values[output_index] = in_values[thread_id_in_block];

		__syncthreads();

		// Swap in_keys and out_keys for the next iteration
		K_t* temp = in_keys;
		in_keys	  = out_keys;
		out_keys  = temp;

		V_t* temp_v = in_values;
		in_values	= out_values;
		out_values	= temp_v;

#if DEBUG_PRINT
		if (DEBUGvery_first && DEBUG_PRINT)
			printf("\n\n");
#endif
	}

	// if (sizeof(K_t) * 8 / RadixSortBlockRadixBits & 1)
	//{
	//	// Copying so the final output is in the input arrays given by the user
	//	in_keys[thread_id_in_block]	  = out_keys[thread_id_in_block];
	//	in_values[thread_id_in_block] = out_values[thread_id_in_block];
	// }

#if DEBUG_PRINT
	if (DEBUGvery_first && DEBUG_PRINT)
		printf("\n\n\n");
#endif
}

#endif
