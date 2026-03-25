/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_WARP_BLOCK_REDUCE_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_WARP_BLOCK_REDUCE_H

#include "Device/includes/Compute/Common/Operators.h"
#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_reduce(T value, int thread_idx = threadIdx.x)
{
	UNROLL_LOOP
	for (int i = 16; i > 0; i >>= 1)
	{
		T n = hippt::warp_shfl_down(value, i);

		value = Operator::apply(value, n);
	}

	return hippt::warp_shfl(value, 0);
}

template <int element_count, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T block_reduce(T value, int thread_idx = threadIdx.x)
{
	int lane				 = thread_idx & 31;
	int warp_index			 = thread_idx >> 5;
	constexpr int warp_count = element_count >> 5;
	static_assert(warp_count << 5 == element_count);

	T warp_reduction = warp_reduce<T, Operator>(value, thread_idx);

	__shared__ T warp_reductions[warp_count];

	if (lane == 0)
		warp_reductions[warp_index] = warp_reduction;

	__syncthreads();

	if (warp_index == 0)
	{
		T warp_reduction_value = Operator::identity;
		if (lane < warp_count)
			warp_reduction_value = warp_reductions[lane];

		warp_reduction_value = warp_reduce<T, Operator>(warp_reduction_value, thread_idx);

		if (lane == 0)
			warp_reductions[0] = warp_reduction_value;
	}

	__syncthreads();

	return warp_reductions[0];
}

#endif
