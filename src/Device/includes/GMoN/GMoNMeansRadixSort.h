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

// Some macros to make that single function work on the CPU and GPU
#ifdef __KERNELCC__
#define GMoNThreadsPerBlock (GMoNComputeMeansKernelThreadBlockSize * GMoNComputeMeansKernelThreadBlockSize)

// Allocating enough shared memory for each thread to store the M keys it's going to need for sorting.
// We multiply everything * 2 by radix sort isn't in place so we need *1 for the input buffer of keys to sort
// and another *1 for the sorted keys
__shared__ unsigned int scratch_memory[GMoNThreadsPerBlock * GMoNMSetsCount * 2];

#define ThreadIndex1D (threadIdx.x + threadIdx.y * blockDim.x)
// The indexing used here tries to avoid bank conflicts
#define SCRATCH_MEMORY_INDEX(input_buffer_index, key_index) (ThreadIndex1D + key_index * GMoNThreadsPerBlock + input_buffer_index * GMoNThreadsPerBlock * GMoNMSetsCount)

#define RETURN_TYPE void

#define INITIAL_LOAD_KEY_IN_INPUT_BUFFER(key_index, value) scratch_memory[SCRATCH_MEMORY_INDEX(0, key_index)] = value

#define READ_KEY(key_index) scratch_memory[SCRATCH_MEMORY_INDEX(input_buffer_index, key_index)]
#define STORE_KEY(key_index, value) scratch_memory[SCRATCH_MEMORY_INDEX(output_buffer_index, key_index)] = value

#else // #ifdef __KERNELCC__

#define SCRATCH_MEMORY_INDEX(input_buffer_index, key_index) (key_index)

#define RETURN_TYPE std::vector<unsigned int>

#define INITIAL_LOAD_KEY_IN_INPUT_BUFFER(key_index, value) keys[key_index] = value

#define READ_KEY(key_index) (keys[SCRATCH_MEMORY_INDEX(42, key_index)])
#define STORE_KEY(key_index, value) scratch_memory[SCRATCH_MEMORY_INDEX(42, key_index)] = value
#endif

HIPRT_HOST_DEVICE HIPRT_INLINE RETURN_TYPE gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution, unsigned short int out_sorted_indices[GMoNMSetsCount])
{
#ifndef __KERNELCC__
	std::vector<unsigned int> keys_vector(GMoNMSetsCount);
	std::vector<unsigned int> scratch_memory_vector(GMoNMSetsCount);

	unsigned int* keys = keys_vector.data();
	unsigned int* scratch_memory = scratch_memory_vector.data();
#else
	unsigned int input_buffer_index = 0;
	unsigned int output_buffer_index = 1;
#endif

	constexpr unsigned int number_of_keys = GMoNMSetsCount;
	
	// Loading in the input scratch memory
	for (int key_index = 0; key_index < number_of_keys; key_index++)
	{
		// Note that this isn't actually the mean, this is just the value of the accumulated samples
		// If we wanted the mean, we would have to divide everyone by the number of samples
		// But dividing everyone by the same value isn't going to change the ordering so we don't have to do
		// that division
		float mean = gmon_sets[key_index * render_resolution.x * render_resolution.y + pixel_index].luminance();

		// Setting the means in the "input buffer"
		INITIAL_LOAD_KEY_IN_INPUT_BUFFER(key_index, *reinterpret_cast<unsigned int*>(&mean));
	}

	// Initializing the sorted indices
	// 
	// The sorted indices are 16 bits.
	// The low 8 bits are the actual sorted indices
	// The high 16 bits are used for internal machinery
	//
	// We only need to initialize the high bits here, the low bits
	// will be overwritten with the sorted indices
	for (int i = 0; i < GMoNMSetsCount; i++)
		out_sorted_indices[i] = i << 8;

	for (int digit = 0; digit < GMoNKeysNbDigitsForRadixSort; digit += GMoNSortRadixSize)
	{
		unsigned int radix_extraction_mask = ((1 << GMoNSortRadixSize) - 1) << digit;
		unsigned int histogram[1 << GMoNSortRadixSize] = { 0 };

		// Computing the histogram for the counting sort
		for (int key_index = 0; key_index < number_of_keys; key_index++)
		{
			unsigned int radix = READ_KEY(key_index) & radix_extraction_mask;
			radix >>= digit;

			histogram[radix]++;
		}

		// Computing the prefix sum for stable counting sort
		for (int i = 1; i < 1 << GMoNSortRadixSize; i++)
			histogram[i] += histogram[i - 1];

		for (int key_index = number_of_keys - 1; key_index >= 0; key_index--)
		{
			unsigned int key = READ_KEY(key_index);
			unsigned int radix = key & radix_extraction_mask;
			radix >>= digit;

			STORE_KEY(--histogram[radix], key);

			// Also sorting a list of indices so that, when returning from this function,
			// we can find from the caller's code which ColorRGB corresponds to the median
			//
			// Clearing the low 8 bits
			out_sorted_indices[histogram[radix]] &= ~0xFF;
			// Setting the sorted index in the low 8 bits
			out_sorted_indices[histogram[radix]] |= (out_sorted_indices[key_index] >> 8);
		}

		// Bookkeeping to prepare the next sorting pass: copying the sorted indices
		// (in the low 8 bits) to the high 8 bits
		for (int i = 0; i < GMoNMSetsCount; i++)
		{
			// Clearing the high 8 bits
			out_sorted_indices[i] &= ~(0xFF << 8);
			// Copying the low 8 bits to the high 8 bits
			out_sorted_indices[i] |= (out_sorted_indices[i] & 0xFF) << 8;
		}

#ifdef __KERNELCC__
		// Swapping the buffer indices on the GPU
		input_buffer_index = input_buffer_index == 0 ? 1 : 0;
		output_buffer_index = output_buffer_index == 0 ? 1 : 0;
#else
		// On the CPU, input/output ping-ponging is just a swap of pointer
		unsigned int* temp = keys;
		keys = scratch_memory;
		scratch_memory = temp;
#endif
	}

#ifndef __KERNELCC__
	// The result is in keys for 32 digit keys
	return keys_vector;
#endif
}

#endif
