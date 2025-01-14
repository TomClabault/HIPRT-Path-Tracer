/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_GMON_RADIX_SORT_H
#define DEVICE_GMON_RADIX_SORT_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions/GMoNOptions.h"
#include "HostDeviceCommon/Math.h"

#ifdef __KERNELCC__

#define GMoNThreadsPerBlock (GMoNComputeMeansKernelThreadBlockSize * GMoNComputeMeansKernelThreadBlockSize)

// Allocating enough shared memory for each thread to store the M keys it's going to need for sorting.
// We multiply everything * 2 by radix sort isn't in place so we need *1 for the input buffer of keys to sort
// and another *1 for the sorted keys
__shared__ unsigned int scratch_memory[GMoNThreadsPerBlock * GMoNMSetsCount * 2];

#define ThreadIndex1D (threadIdx.x + threadIdx.y * blockDim.x)
// The indexing used here tries to avoid bank conflicts
#define SCRATCH_MEMORY_INDEX(input_buffer_index, key_index) (ThreadIndex1D + key_index * GMoNThreadsPerBlock + input_buffer_index * GMoNThreadsPerBlock * GMoNMSetsCount)

HIPRT_HOST_DEVICE HIPRT_INLINE void gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
	unsigned int input_buffer_index = 0;
	unsigned int output_buffer_index = 1;
	
	// Loading in shared memory
	for (int set_index = 0; set_index < GMoNMSetsCount; set_index++)
	{
		// Note that this isn't actually the mean, this is just the value of the accumulated samples
		// If we wanted the mean, we would have to divide everyone by the number of samples
		// But dividing everyone by the same value isn't going to change the ordering so we don't have to do
		// that division
		float mean = gmon_sets[set_index * render_resolution.x * render_resolution.y + pixel_index].luminance();

		// Setting the means in the "input buffer" in shared memory
		scratch_memory[SCRATCH_MEMORY_INDEX(0, set_index)] = *reinterpret_cast<unsigned int*>(&mean);
	}


	for (int digit = 0; digit < GMoNKeysNbDigitsForRadixSort; digit++)
	{
		unsigned int nb_zeroes = 0;
		unsigned int nb_ones = 0;

		for (int key = 0; key < GMoNMSetsCount; key++)
		{
			if (scratch_memory[SCRATCH_MEMORY_INDEX(input_buffer_index, key)] & (1 << digit))
				nb_ones++;
			else
				nb_zeroes++;
		}

		unsigned int prefix_sum_0 = nb_zeroes;
		unsigned int prefix_sum_1 = nb_zeroes + nb_ones;

		for (int key_index = GMoNMSetsCount - 1; key_index >= 0; key_index--)
		{
			unsigned int key = scratch_memory[SCRATCH_MEMORY_INDEX(input_buffer_index, key_index)];
			if (key & (1 << digit))
				// The key has a 1 for digit
				scratch_memory[SCRATCH_MEMORY_INDEX(output_buffer_index, --prefix_sum_1)] = key;
			else
				scratch_memory[SCRATCH_MEMORY_INDEX(output_buffer_index, --prefix_sum_0)] = key;
		}

		input_buffer_index = input_buffer_index == 0 ? 1 : 0;
		output_buffer_index = output_buffer_index == 0 ? 1 : 0;
	}
}

#else

HIPRT_HOST_DEVICE HIPRT_INLINE std::vector<unsigned int> gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
	std::vector<unsigned int> keys_vector(GMoNMSetsCount);
	std::vector<unsigned int> scratch_memory_vector(GMoNMSetsCount);

	unsigned int* keys = keys_vector.data();
	unsigned int* scratch_memory = scratch_memory_vector.data();

	// Loading the means in the keys vectors
	for (int i = 0; i < GMoNMSetsCount; i++)
	{
		// Note that this isn't actually the mean, this is just the value of the accumulated samples
		// If we wanted the mean, we would have to divide everyone by the number of samples
		// But dividing everyone by the same value isn't going to change the ordering so we don't have to do
		// that division	
		float mean = gmon_sets[i * render_resolution.x * render_resolution.y + pixel_index].luminance();

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
