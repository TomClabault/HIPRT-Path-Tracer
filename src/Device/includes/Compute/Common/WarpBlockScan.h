// Register-based warp inclusive scan

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

template <typename T>
HIPRT_DEVICE int warp_scan_inclusive(T val, int thread_idx = threadIdx.x)
{
	unsigned int lane  = thread_idx & 31;
	int inclusive_scan = val;

	UNROLL_LOOP
	for (int i = 1; i <= 16; i *= 2)
	{
		int n = hippt::warp_shfl_up(inclusive_scan, i);

		if (lane >= i)
			inclusive_scan += n;
	}

	return inclusive_scan;
}

template <typename T>
HIPRT_DEVICE int warp_scan_exclusive(T val, int thread_idx = threadIdx.x)
{
	return warp_scan_inclusive(val, thread_idx) - val;
}

/**
 * element_count must be a multiple of 32 and element count must be equal to the number of threads in the block
 */
template <int element_count, typename T>
HIPRT_DEVICE int block_scan_inclusive(T val, int thread_idx = threadIdx.x)
{
	int lane				 = thread_idx & 31;
	int warp_index			 = thread_idx >> 5;
	constexpr int warp_count = element_count >> 5;
	static_assert(warp_count << 5 == element_count);

	__shared__ int warp_sums[warp_count];

	int warp_sum = warp_scan_inclusive(val, thread_idx);

	if (lane == 31)
		warp_sums[warp_index] = warp_sum;
	__syncthreads();

	if (warp_index == 0)
	{
		int my_warp_sum = 0;
		if (lane < warp_count)
			my_warp_sum = warp_sums[lane];

		int warp_sums_scan = warp_scan_inclusive(my_warp_sum, thread_idx);

		if (lane < warp_count)
			warp_sums[lane] = warp_sums_scan;
	}

	__syncthreads();

	int block_sum = warp_sum;
	if (warp_index > 0)
		block_sum += warp_sums[warp_index - 1];

	return block_sum;
}

template <int element_count, typename T>
HIPRT_DEVICE int block_scan_exclusive(T val, int thread_idx = threadIdx.x)
{
	return block_scan_inclusive<element_count>(val, thread_idx) - val;
}
