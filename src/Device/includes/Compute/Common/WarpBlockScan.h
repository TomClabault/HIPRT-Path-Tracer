/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_WARP_BLOCK_SCAN_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_WARP_BLOCK_SCAN_H

#include "Device/includes/Compute/Common/Operators.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_prefix_scan_inclusive(T val, int thread_idx = threadIdx.x)
{
	unsigned int lane = thread_idx & 31;
	T inclusive_scan  = val;

	UNROLL_LOOP
	for (int i = 1; i <= 16; i <<= 1)
	{
		T n = hippt::warp_shfl_up(inclusive_scan, i);

		if (lane >= i)
			inclusive_scan = Operator::apply(n, inclusive_scan);
	}

	return inclusive_scan;
}

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_prefix_scan_exclusive(T val, int thread_idx = threadIdx.x)
{
	const unsigned int lane = thread_idx & 31;

	// Shift right by one lane, insert identity at lane 0
	T val_shifted = hippt::warp_shfl_up(val, 1);
	val_shifted	  = lane == 0 ? 0 : val_shifted;

	return warp_prefix_scan_inclusive<T, Operator>(val_shifted, thread_idx);
}

// Register-based warp inclusive scan
// Flag here is an unsigned int type so that it can be used in shfl instructions but its value should be either 0 or 1.
template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_segmented_scan_inclusive(T val, unsigned int flag, unsigned int& out_propagated_flag)
{
	unsigned int lane = threadIdx.x & 31;

	UNROLL_LOOP
	for (int i = 1; i <= 16; i *= 2)
	{
		T n			   = hippt::warp_shfl_up(val, i);
		unsigned int f = hippt::warp_shfl_up(flag, i);

		if (lane >= i)
		{
			// If this is a segment start (flag == true), we don't want to add the value from the previous lane, so we just keep our 'val'
			val = flag ? val : Operator::apply(n, val);
			flag |= f;
		}
	}

	out_propagated_flag = flag;

	return val;
}

/**
 * element_count must be a multiple of 32 and element count must be equal to the number of threads in the block
 */
template <int element_count, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T block_prefix_scan_inclusive(T val, int thread_idx = threadIdx.x)
{
	int lane				 = thread_idx & 31;
	int warp_index			 = thread_idx >> 5;
	constexpr int warp_count = element_count >> 5;
	static_assert(warp_count << 5 == element_count);

	__shared__ T warp_scans[warp_count];

	T warp_sum = warp_prefix_scan_inclusive<T, Operator>(val, thread_idx);

	if (lane == 31)
		warp_scans[warp_index] = warp_sum;
	__syncthreads();

	if (warp_index == 0)
	{
		T my_warp_scan = Operator::identity;
		if (lane < warp_count)
			my_warp_scan = warp_scans[lane];

		T warp_scans_scan = warp_prefix_scan_inclusive<T, Operator>(my_warp_scan, thread_idx);

		if (lane < warp_count)
			warp_scans[lane] = warp_scans_scan;
	}

	__syncthreads();

	T block_scan = warp_sum;
	if (warp_index > 0)
		block_scan = Operator::apply(warp_scans[warp_index - 1], block_scan);

	return block_scan;
}

template <int element_count, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T block_prefix_scan_exclusive(T val, int thread_idx = threadIdx.x)
{
	T inclusive_scanned = block_prefix_scan_inclusive<element_count, T, Operator>(val, thread_idx);
	T exclusive_scanned = inclusive_scanned - val;

	return exclusive_scanned;
}

#endif
