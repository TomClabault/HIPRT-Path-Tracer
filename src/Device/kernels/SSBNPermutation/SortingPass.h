/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_SSBN_PERMUTATION_SORTING_PASS_H
#define KERNELS_SSBN_PERMUTATION_SORTING_PASS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/KernelOptions/SSBNPermutationOptions.h"
#include "HostDeviceCommon/RenderData.h"

template <typename T>
HIPRT_DEVICE void bubble_sort(T keys[SSBNPermutationBlockSize * SSBNPermutationBlockSize], int2_t values[SSBNPermutationBlockSize * SSBNPermutationBlockSize])
{
	int n = SSBNPermutationBlockSize * SSBNPermutationBlockSize;

	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - i - 1; j++)
		{
			if (keys[j] > keys[j + 1])
			{
				T temp_key	= keys[j];
				keys[j]		= keys[j + 1];
				keys[j + 1] = temp_key;

				int2_t temp_value = values[j];
				values[j]		  = values[j + 1];
				values[j + 1]	  = temp_value;
			}
		}
	}
}

GLOBAL_KERNEL_SIGNATURE(void)
SSBNPermutationSortingPass(HIPRTRenderData render_data,
						   unsigned char* __restrict__ blue_noise_dither_texture_buffer,
						   unsigned int* __restrict__ in_seeds_to_sort,
						   unsigned int* __restrict__ out_sorted_seeds_buffer)
{
	unsigned int resolution_x = render_data.render_settings.render_resolution.x;
	unsigned int resolution_y = render_data.render_settings.render_resolution.y;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int in_block_x = threadIdx.x;
	unsigned int in_block_y = threadIdx.y;

	unsigned int pixel_index = x + y * resolution_x;
	if (pixel_index == 0)
		printf("Block size: %d\n", SSBNPermutationBlockSize);
	unsigned int thread_index_in_block = in_block_x + in_block_y * SSBNPermutationBlockSize;

	bool thread_valid = x < resolution_x && y < resolution_y;

	__shared__ unsigned char input_blue_noise[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ int2_t input_blue_noise_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ float input_pixel_luminance[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ int2_t input_pixel_luminance_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];

	constexpr float golden_ratio_2d = 1.3247179572f;
	const int blue_noise_offset_x	= 0; // (int)(1.0f / golden_ratio_2d * resolution_x * render_data.render_settings.sample_number) % SSBNPermutationBlockSize;
	const int blue_noise_offset_y	= 0; // (int)(1.0f / (golden_ratio_2d * golden_ratio_2d) * resolution_y * render_data.render_settings.sample_number) %
										 // SSBNPermutationBlockSize;

	unsigned int blue_noise_index_x = (x + blue_noise_offset_x) % SSBNPermutationBlueNoiseTextureSize;
	unsigned int blue_noise_index_y = (y + blue_noise_offset_y) % SSBNPermutationBlueNoiseTextureSize;

	if (thread_valid)
	{
		input_blue_noise[thread_index_in_block] =
								blue_noise_dither_texture_buffer[blue_noise_index_x + blue_noise_index_y * SSBNPermutationBlueNoiseTextureSize];
		input_blue_noise_coordinates[thread_index_in_block] = make_int2(in_block_x, in_block_y);

		input_pixel_luminance[thread_index_in_block]			 = render_data.buffers.accumulated_ray_colors[pixel_index].luminance();
		input_pixel_luminance_coordinates[thread_index_in_block] = make_int2(in_block_x, in_block_y);
	}
	else
	{
		input_blue_noise[thread_index_in_block]				= 255;
		input_blue_noise_coordinates[thread_index_in_block] = make_int2(-1, -1);

		input_pixel_luminance[thread_index_in_block]			 = 1.0e30f;
		input_pixel_luminance_coordinates[thread_index_in_block] = make_int2(-1, -1);
	}

	__syncthreads();

	if (in_block_x == 0 && in_block_y == 0)
	{
		bubble_sort(input_blue_noise, input_blue_noise_coordinates);
		bubble_sort(input_pixel_luminance, input_pixel_luminance_coordinates);
	}

	__syncthreads();

	unsigned int blue_noise_block_sorted_index = input_blue_noise_coordinates[thread_index_in_block].x +
												 input_blue_noise_coordinates[thread_index_in_block].y * SSBNPermutationBlockSize;

	unsigned int block_start_x			   = blockIdx.x * SSBNPermutationBlockSize;
	unsigned int block_start_y			   = blockIdx.y * SSBNPermutationBlockSize;
	unsigned int block_start_global_offset = block_start_x + block_start_y * resolution_x;

	int2_t luminance_coords					   = input_pixel_luminance_coordinates[thread_index_in_block];
	unsigned int luminance_global_sorted_index = block_start_global_offset + input_pixel_luminance_coordinates[thread_index_in_block].x +
												 input_pixel_luminance_coordinates[thread_index_in_block].y * resolution_x;

	__shared__ unsigned int sorted_seeds[SSBNPermutationBlockSize * SSBNPermutationBlockSize];

	if (luminance_coords.x != -1 && luminance_coords.y != -1)
		sorted_seeds[blue_noise_block_sorted_index] = in_seeds_to_sort[luminance_global_sorted_index];

	__syncthreads();

	if (thread_valid)
		out_sorted_seeds_buffer[pixel_index] = sorted_seeds[thread_index_in_block];
}

#endif
