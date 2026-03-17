/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_SSBN_PERMUTATION_SORTING_PASS_H
#define KERNELS_SSBN_PERMUTATION_SORTING_PASS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/SSBNPermutation/SSBNPermutationCommon.h"
#include "HostDeviceCommon/KernelOptions/SSBNPermutationOptions.h"
#include "HostDeviceCommon/RenderData.h"

template <typename K_t, typename V_t>
HIPRT_DEVICE void bubble_sort(K_t keys[SSBNPermutationBlockSize * SSBNPermutationBlockSize], V_t values[SSBNPermutationBlockSize * SSBNPermutationBlockSize])
{
	int n = SSBNPermutationBlockSize * SSBNPermutationBlockSize;

	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - i - 1; j++)
		{
			if (keys[j] > keys[j + 1])
			{
				K_t temp_key = keys[j];
				keys[j]		 = keys[j + 1];
				keys[j + 1]	 = temp_key;

				V_t temp_value = values[j];
				values[j]	   = values[j + 1];
				values[j + 1]  = temp_value;
			}
		}
	}
}

GLOBAL_KERNEL_SIGNATURE(void)
SSBNPermutationSortingPass(HIPRTRenderData render_data,
						   unsigned char* __restrict__ blue_noise_dither_texture_buffer,
						   unsigned int blue_noise_texture_width,
						   unsigned int blue_noise_texture_height,
						   unsigned int* __restrict__ in_seeds_to_sort,
						   unsigned int* __restrict__ out_sorted_seeds_buffer)
{
	int resolution_x		= render_data.render_settings.render_resolution.x;
	int resolution_y		= render_data.render_settings.render_resolution.y;
	int padded_resolution_x = (resolution_x + blue_noise_texture_width - 1) / blue_noise_texture_width * blue_noise_texture_width;
	int padded_resolution_y = (resolution_y + blue_noise_texture_height - 1) / blue_noise_texture_height * blue_noise_texture_height;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int in_block_x = threadIdx.x;
	int in_block_y = threadIdx.y;

	int pixel_index			  = x + y * resolution_x;
	int thread_index_in_block = in_block_x + in_block_y * blockDim.x;

	bool thread_valid = x < resolution_x && y < resolution_y;

	__shared__ short int input_blue_noise[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ short2_t input_blue_noise_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ fp16 input_pixel_luminance[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ short2_t input_pixel_luminance_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ unsigned int sorted_seeds[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	sorted_seeds[thread_index_in_block] = 0;

	int blue_noise_offset_x = 0, blue_noise_offset_y = 0;
	get_blue_noise_texture_offset(blue_noise_texture_width, blue_noise_texture_height, render_data.render_settings.sample_number, blue_noise_offset_x,
								  blue_noise_offset_y);

	int blue_noise_index_x = (x + blue_noise_offset_x) % blue_noise_texture_width;
	int blue_noise_index_y = (y + blue_noise_offset_y) % blue_noise_texture_height;

	input_blue_noise[thread_index_in_block]				= blue_noise_dither_texture_buffer[blue_noise_index_x + blue_noise_index_y * blue_noise_texture_width];
	input_blue_noise_coordinates[thread_index_in_block] = make_short2(in_block_x, in_block_y);

	// First sample and our thread is in the padded area of the seed buffer, reading from mirrored values for the luminance
	// For pixels that are in-screen, the mirrored_pixel_index is just the pixel index
	int mirrored_at_edge_x	 = x >= resolution_x ? (resolution_x - 1 - (x - resolution_x)) : x;
	int mirrored_at_edge_y	 = y >= resolution_y ? (resolution_y - 1 - (y - resolution_y)) : y;
	int mirrored_pixel_index = mirrored_at_edge_x + mirrored_at_edge_y * resolution_x;

	input_pixel_luminance[thread_index_in_block]			 = render_data.buffers.last_frame_ray_colors[mirrored_pixel_index].luminance();
	input_pixel_luminance_coordinates[thread_index_in_block] = make_short2(in_block_x, in_block_y);

	__syncthreads();

	if (in_block_x == 0 && in_block_y == 0)
	{
		bubble_sort(input_blue_noise, input_blue_noise_coordinates);
		bubble_sort(input_pixel_luminance, input_pixel_luminance_coordinates);
	}

	__syncthreads();

	int blue_noise_block_sorted_index = input_blue_noise_coordinates[thread_index_in_block].x +
										input_blue_noise_coordinates[thread_index_in_block].y * SSBNPermutationBlockSize;

	unsigned int block_start_x			   = blockIdx.x * SSBNPermutationBlockSize;
	unsigned int block_start_y			   = blockIdx.y * SSBNPermutationBlockSize;
	unsigned int block_start_global_offset = block_start_x + block_start_y * padded_resolution_x;

	short2_t luminance_coords		  = input_pixel_luminance_coordinates[thread_index_in_block];
	int luminance_global_sorted_index = block_start_global_offset + input_pixel_luminance_coordinates[thread_index_in_block].x +
										input_pixel_luminance_coordinates[thread_index_in_block].y * padded_resolution_x;

	sorted_seeds[blue_noise_block_sorted_index] = in_seeds_to_sort[luminance_global_sorted_index];

	__syncthreads();

	// Full pixel index that can go in the padded area of the seeds buffer
	int full_pixel_index = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * padded_resolution_x;

	out_sorted_seeds_buffer[full_pixel_index] = sorted_seeds[thread_index_in_block];
}

#endif
