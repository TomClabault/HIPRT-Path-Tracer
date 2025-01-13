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

__shared__ unsigned int scratch_memory[GMON_MEANS_RADIX_SORT_KERNEL_BLOCK_SIZE * GMON_MEANS_RADIX_SORT_KERNEL_BLOCK_SIZE * GMON_M_SETS_COUNT * 2];
#define SCRATCH_MEMORY_INDEX(input_buffer_index, index) ((index * 32 + threadIdx.x) * (input_buffer_index + 1))

template <unsigned int nbDigits>
HIPRT_HOST_DEVICE HIPRT_INLINE void gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
	bool input_buffer_index = false;

	// Loading in shared memory
	for (int i = 0; i < GMON_M_SETS_COUNT; i++)
	{
		float mean = gmon_sets[i * render_resolution.x * render_resolution.y + pixel_index].luminance() / (sample_number / GMON_M_SETS_COUNT);

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

#include <iomanip>

HIPRT_HOST_DEVICE HIPRT_INLINE std::vector<unsigned int> gmon_means_radix_sort(ColorRGB32F* gmon_sets, uint32_t pixel_index, unsigned int sample_number, int2 render_resolution)
{
	std::vector<unsigned int> keys_vector(GMON_M_SETS_COUNT);
	std::vector<unsigned int> scratch_memory_vector(GMON_M_SETS_COUNT);

	unsigned int* keys = keys_vector.data();
	unsigned int* scratch_memory = scratch_memory_vector.data();

	// Loading in the keys vectors
	for (int i = 0; i < GMON_M_SETS_COUNT; i++)
	{
		//std::cout << std::fixed << std::setprecision(10) << "[" << gmon_sets[i * render_resolution.x * render_resolution.y + pixel_index] << "], ";
		float mean = gmon_sets[i * render_resolution.x * render_resolution.y + pixel_index].luminance() / (sample_number / GMON_M_SETS_COUNT);
		//std::cout << std::fixed << std::setprecision(10) <<  mean << std::endl;

		keys[i] = *reinterpret_cast<unsigned int*>(&mean);
	}
	//std::cout << std::endl;

	for (int digit = 0; digit < GMON_NB_KEYS_DIGITS; digit++)
	{
		unsigned int nb_zeroes = 0;
		unsigned int nb_ones = 0;

		for (int key = 0; key < GMON_M_SETS_COUNT; key++)
		{
			if (keys[key] & (1 << digit))
				nb_ones++;
			else
				nb_zeroes++;
		}

		unsigned int prefix_sum_0 = nb_zeroes;
		unsigned int prefix_sum_1 = nb_zeroes + nb_ones;

		for (int key = GMON_M_SETS_COUNT - 1; key >= 0; key--)
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
