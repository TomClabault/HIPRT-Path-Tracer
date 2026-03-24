// Register-based warp inclusive scan

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

#include <limits>

template <typename T>
struct OperatorSum
{
	static constexpr T identity = T(0);

	HIPRT_DEVICE static T apply(T a, T b)
	{
		return a + b;
	}
};

template <typename T>
struct OperatorMin
{
	static constexpr T identity = std::numeric_limits<T>::max();

	HIPRT_DEVICE static T apply(T a, T b)
	{
		return hippt::min(a, b);
	}
};

template <typename T>
struct OperatorMax
{
	static constexpr T identity = std::numeric_limits<T>::lowest();

	HIPRT_DEVICE static T apply(T a, T b)
	{
		return hippt::max(a, b);
	}
};

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE int warp_scan_inclusive(T val, int thread_idx = threadIdx.x)
{
	unsigned int lane  = thread_idx & 31;
	int inclusive_scan = val;

	UNROLL_LOOP
	for (int i = 1; i <= 16; i <<= 1)
	{
		int n = hippt::warp_shfl_up(inclusive_scan, i);

		if (lane >= i)
			inclusive_scan = Operator::apply(n, inclusive_scan);
	}

	return inclusive_scan;
}

template <typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE T warp_scan_exclusive(T val, int thread_idx = threadIdx.x)
{
	const unsigned int lane = thread_idx & 31;

	// Shift right by one lane, insert identity at lane 0
	T val_shifted = hippt::warp_shfl_up(val, 1);
	val_shifted	  = lane == 0 ? 0 : val_shifted;

	return warp_scan_inclusive<T, Operator>(val_shifted, thread_idx);
}

/**
 * element_count must be a multiple of 32 and element count must be equal to the number of threads in the block
 */
template <int element_count, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE int block_scan_inclusive(T val, int thread_idx = threadIdx.x)
{
	int lane				 = thread_idx & 31;
	int warp_index			 = thread_idx >> 5;
	constexpr int warp_count = element_count >> 5;
	static_assert(warp_count << 5 == element_count);

	__shared__ int warp_sums[warp_count];

	int warp_sum = warp_scan_inclusive<T, Operator>(val, thread_idx);

	if (lane == 31)
		warp_sums[warp_index] = warp_sum;
	__syncthreads();

	if (warp_index == 0)
	{
		int my_warp_sum = 0;
		if (lane < warp_count)
			my_warp_sum = warp_sums[lane];

		int warp_sums_scan = warp_scan_inclusive<T, Operator>(my_warp_sum, thread_idx);

		if (lane < warp_count)
			warp_sums[lane] = warp_sums_scan;
	}

	__syncthreads();

	int block_sum = warp_sum;
	if (warp_index > 0)
		block_sum = Operator::apply(warp_sums[warp_index - 1], block_sum);

	return block_sum;
}

template <int element_count, typename T, typename Operator = OperatorSum<T>>
HIPRT_DEVICE int block_scan_exclusive(T val, int thread_idx = threadIdx.x)
{
	const unsigned int lane = thread_idx & 31;

	// Shift right by one lane, insert identity at lane 0
	T val_shifted = hippt::warp_shfl_up(val, 1);
	val_shifted	  = lane == 0 ? 0 : val_shifted;

	return block_scan_inclusive<element_count, T, Operator>(val_shifted, thread_idx);
}
