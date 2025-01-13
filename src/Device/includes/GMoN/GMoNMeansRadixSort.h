/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GMON_RADIX_SORT_H
#define DEVICE_GMON_RADIX_SORT_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Math.h"

#ifdef __KERNELCC__

#define GMoNThreadsPerBlock (GMoNComputeMeansKernelThreadBlockSize * GMoNComputeMeansKernelThreadBlockSize)

// Allocating enough shared memory for each thread to store the M keys it's going to need for sorting.
// We multiply everything * 2 by radix sort isn't in place so we need *1 for the input buffer of keys to sort
// and another *1 for the sorted keys
__shared__ unsigned int scratch_memory[GMoNThreadsPerBlock * GMoNMSetsCount * 2];

#define ThreadIndex2DTo1D (threadIdx.x + threadIdx.y * blockDim.x)
#define SCRATCH_MEMORY_INDEX(input_buffer_index, key_index) (ThreadIndex2DTo1D + key_index * GMoNThreadsPerBlock + input_buffer_index * GMoNThreadsPerBlock * GMoNMSetsCount)

template <unsigned int nbDigits>
HIPRT_HOST_DEVICE HIPRT_INLINE void gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
	bool input_buffer_index = false;

	// Loading in shared memory
	for (int i = 0; i < GMoNMSetsCount; i++)
	{
		float mean = gmon_sets[i * render_resolution.x * render_resolution.y + pixel_index].luminance() / (sample_number / GMoNMSetsCount);

		scratch_memory[SCRATCH_MEMORY_INDEX(0, i)] = *reinterpret_cast<unsigned int*>(&mean);
	}

	for (int digit = 0; digit < nbDigits; digit++)
	{
		unsigned int nb_zeroes = 0;
		unsigned int nb_ones = 0;

		for (int key = 0; key < number_keys; key++)
		{
			if (scratch_memory[SCRATCH_MEMORY_INDEX(input_buffer_index, key)] & (1 << digit))
				nb_ones++;
			else
				nb_zeroes++;
		}

		unsigned int prefix_sum_0 = nb_zeroes;
		unsigned int prefix_sum_1 = nb_zeroes + nb_ones;

		for (int key = number_keys - 1; key >= 0; key--)
		{
			if (keys[key] & (1 << digit))
				// The key has a 1 for digit
				scratch_memory[SCRATCH_MEMORY_INDEX(input_buffer_index, --prefix_sum_1)] = keys[key];
			else
				scratch_memory[SCRATCH_MEMORY_INDEX(input_buffer_index, --prefix_sum_0)] = keys[key];
		}

		input_buffer_index = !input_buffer_index;
	}
}

#else

HIPRT_HOST_DEVICE HIPRT_INLINE std::vector<unsigned int> gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
	std::vector<unsigned int> keys_vector(GMoNMSetsCount);
	std::vector<unsigned int> scratch_memory_vector(GMoNMSetsCount);

	unsigned int* keys = keys_vector.data();
	unsigned int* scratch_memory = scratch_memory_vector.data();

	// samples_per_set is an integer because we're only getting in this part of the codebase when GMoN has
	// accumulated one more sample per set i.e. when the number of samples rendered by the renderer so far
	// is a multiple of GMoNMSetsCount
	unsigned int samples_per_set = sample_number / GMoNMSetsCount;

	// Loading the means in the keys vectors
	for (int i = 0; i < GMoNMSetsCount; i++)
	{
		float mean = gmon_sets[i * render_resolution.x * render_resolution.y + pixel_index].luminance() / samples_per_set;

		keys[i] = *reinterpret_cast<unsigned int*>(&mean);
	}

	for (int digit = 0; digit < GMoNKeysNbDigitsForRadixSort; digit++)
	{
		unsigned int nb_zeroes = 0;
		unsigned int nb_ones = 0;

		for (int key = 0; key < GMoNMSetsCount; key++)
		{
			if (keys[key] & (1 << digit))
				nb_ones++;
			else
				nb_zeroes++;
		}

		unsigned int prefix_sum_0 = nb_zeroes;
		unsigned int prefix_sum_1 = nb_zeroes + nb_ones;

		for (int key = GMoNMSetsCount - 1; key >= 0; key--)
		{
			if (keys[key] & (1 << digit))
				// The key has a 1 for digit
				scratch_memory[--prefix_sum_1] = keys[key];
			else
				scratch_memory[--prefix_sum_0] = keys[key];
		}

		unsigned int* temp = keys;
		keys = scratch_memory;
		scratch_memory = temp;
	}

	// The result is in keys for 32 digit keys
	return keys_vector;
}
#endif // CPU/GPU Radix sort implementation

#endif
