/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_WARP_BLOCK_REDUCE_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_WARP_BLOCK_REDUCE_H

#include "Device/includes/Compute/Common/Operators.h"
#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_reduce(T value)
{
	UNROLL_LOOP
	for (int i = 16; i > 0; i >>= 1)
	{
		T n = hippt::warp_shfl_down(value, i);

		value = Operator::apply(value, n);
	}

	return hippt::warp_shfl(value, 0);
}

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_segmented_reduce(T value, unsigned int flag, unsigned int& out_propagated_flag)
{
	unsigned int lane = threadIdx.x & 31;

	UNROLL_LOOP
	for (int i = 1; i <= 16; i <<= 1)
	{
		T n			   = hippt::warp_shfl_up(value, i);
		unsigned int f = hippt::warp_shfl_up(flag, i);

		if (lane >= i)
		{
			value = flag ? value : Operator::apply(value, n);
			flag |= f;
		}
	}

	out_propagated_flag = flag;

	return value;
}

template <int blockSize, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T block_reduce(T value, int thread_idx = threadIdx.x)
{
	int lane				 = thread_idx & 31;
	int warp_index			 = thread_idx >> 5;
	constexpr int warp_count = blockSize >> 5;
	static_assert(warp_count << 5 == blockSize);

	T warp_reduction = warp_reduce<T, Operator>(value);

	__shared__ T warp_reductions[warp_count];

	if (lane == 0)
		warp_reductions[warp_index] = warp_reduction;

	__syncthreads();

	if (warp_index == 0)
	{
		T warp_reduction_value = Operator::identity;
		if (lane < warp_count)
			warp_reduction_value = warp_reductions[lane];

		warp_reduction_value = warp_reduce<T, Operator>(warp_reduction_value);

		if (lane == 0)
			warp_reductions[0] = warp_reduction_value;
	}

	__syncthreads();

	return warp_reductions[0];
}

template <int blockSize, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T block_segmented_reduce(T thread_input_value, unsigned int flag, int tid)
{
	unsigned int lane_id			  = tid % 32;
	unsigned int warp_id			  = tid >> 5;
	constexpr unsigned int warp_count = blockSize / 32;

	// Per-warp scan
	unsigned int propagated_flag;
	T warp_reduced = warp_segmented_reduce(thread_input_value, flag, propagated_flag);

	__shared__ T smem_warp_last_segment_reductions[blockSize / 32];
	__shared__ unsigned int smem_warp_flags[blockSize / 32];

	if (lane_id == 31)
	{
		smem_warp_last_segment_reductions[warp_id] = warp_reduced;
		smem_warp_flags[warp_id]				   = propagated_flag;
	}

	__syncthreads();

	if (warp_id == 0)
	{
		T my_warp_reduction		  = 0;
		unsigned int my_warp_flag = 0;

		if (lane_id < warp_count)
		{
			my_warp_reduction = smem_warp_last_segment_reductions[lane_id];
			my_warp_flag	  = smem_warp_flags[lane_id];
		}

		unsigned int dummy_flag;
		T warp_wide_reduction = warp_segmented_scan_inclusive(my_warp_reduction, my_warp_flag, dummy_flag);

		smem_warp_last_segment_reductions[lane_id] = warp_wide_reduction;
	}

	__syncthreads();

	if (warp_id > 0)
	{
		// Add the reduction of the previous warps to the current warp reduction
		bool warp_is_open						   = hippt::warp_shfl(flag, 0) == 0; // Whether or not the warp begins with an open flag
		bool no_flag_up_to_current_lane			   = propagated_flag == 0;
		bool warp_is_continuation_of_previous_warp = warp_is_open && no_flag_up_to_current_lane;
		if (warp_is_continuation_of_previous_warp)
			warp_reduced = Operator::apply(warp_reduced, smem_warp_last_segment_reductions[warp_id - 1]);
	}

	return warp_reduced;
}

#endif
