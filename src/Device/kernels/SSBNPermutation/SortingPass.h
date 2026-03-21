/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_SSBN_PERMUTATION_SORTING_PASS_H
#define KERNELS_SSBN_PERMUTATION_SORTING_PASS_H

#include "Device/includes/Compute/RadixSortBlock.h"
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
						   unsigned int* __restrict__ out_sorted_seeds_buffer,
						   const int* __restrict__ in_hash_grid_cell_offsets_buffer)
{
	int resolution_x		= render_data.render_settings.render_resolution.x;
	int resolution_y		= render_data.render_settings.render_resolution.y;
	int padded_resolution_x = (resolution_x + blue_noise_texture_width - 1) / blue_noise_texture_width * blue_noise_texture_width;
	int padded_resolution_y = (resolution_y + blue_noise_texture_height - 1) / blue_noise_texture_height * blue_noise_texture_height;

	int thread_index_in_block = threadIdx.x;

	int hash_grid_cell_offset = in_hash_grid_cell_offsets_buffer[blockIdx.x];
	int end_of_current_hash_grid_cell;
	if (blockIdx.x == gridDim.x - 1)
		end_of_current_hash_grid_cell = resolution_x * resolution_y;
	else
		end_of_current_hash_grid_cell = in_hash_grid_cell_offsets_buffer[blockIdx.x + 1];

	unsigned int pixel_hash = 0;
	int pixel_x				= 0;
	int pixel_y				= 0;
	if (hash_grid_cell_offset + thread_index_in_block < end_of_current_hash_grid_cell)
	{
		uint3_t pixel_hash_grid_data = render_data.ssbn_settings.screen_space_hash_grid[hash_grid_cell_offset + thread_index_in_block];
		pixel_hash					 = pixel_hash_grid_data.x;
		pixel_x						 = pixel_hash_grid_data.y;
		pixel_y						 = pixel_hash_grid_data.z;
	}

	bool thread_valid	  = pixel_x < resolution_x && pixel_y < resolution_y;
	bool valid_input_data = pixel_hash != 0;

	__shared__ short int input_blue_noise[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ short int input_blue_noise_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	// input_pixel_luminance are fp16 but with bits reinterpreted as unsigned short int for the radix sort. We don't have negative luminance or NaNs/special
	// values so reinterpreting is fine
	__shared__ unsigned short int input_pixel_luminance[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	// These can be short int with linear index instead of short2
	__shared__ short2_t input_pixel_luminance_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ unsigned int sorted_seeds[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	sorted_seeds[thread_index_in_block] = 0;

	int blue_noise_offset_x = 0, blue_noise_offset_y = 0;
	get_blue_noise_texture_offset(blue_noise_texture_width, blue_noise_texture_height, render_data.render_settings.sample_number, blue_noise_offset_x,
								  blue_noise_offset_y);

	int blue_noise_index_x = (pixel_x + blue_noise_offset_x) % blue_noise_texture_width;
	int blue_noise_index_y = (pixel_y + blue_noise_offset_y) % blue_noise_texture_height;

	if (valid_input_data && thread_valid)
	{
		input_blue_noise[thread_index_in_block] = blue_noise_dither_texture_buffer[blue_noise_index_x + blue_noise_index_y * blue_noise_texture_width];

		int screen_space_x									= thread_index_in_block % SSBNPermutationBlockSize;
		int screen_space_y									= thread_index_in_block / SSBNPermutationBlockSize;
		input_blue_noise_coordinates[thread_index_in_block] = screen_space_x + screen_space_y * SSBNPermutationBlockSize;
	}
	else
	{
		input_blue_noise[thread_index_in_block]				= 32767;
		input_blue_noise_coordinates[thread_index_in_block] = -1;
	}

	// First sample and our thread is in the padded area of the seed buffer, reading from mirrored values for the luminance
	// For pixels that are in-screen, the mirrored_pixel_index is just the pixel index
	int mirrored_at_edge_x	 = pixel_x >= resolution_x ? (resolution_x - 1 - (pixel_x - resolution_x)) : pixel_x;
	int mirrored_at_edge_y	 = pixel_y >= resolution_y ? (resolution_y - 1 - (pixel_y - resolution_y)) : pixel_y;
	int mirrored_pixel_index = mirrored_at_edge_x + mirrored_at_edge_y * resolution_x;

	if (!valid_input_data || !thread_valid)
	{
		input_pixel_luminance[thread_index_in_block]			 = (unsigned short int)-1;
		input_pixel_luminance_coordinates[thread_index_in_block] = make_short2(-1, -1);
	}
	else
	{
		fp16 luminance								 = static_cast<fp16>(render_data.buffers.last_frame_ray_colors[mirrored_pixel_index].luminance());
		input_pixel_luminance[thread_index_in_block] = hippt::half_as_ushort(luminance);
		input_pixel_luminance_coordinates[thread_index_in_block] = make_short2(pixel_x, pixel_y);
	}

	__syncthreads();

	radix_threadblock_sort<SSBNPermutationBlockSize * SSBNPermutationBlockSize>(input_blue_noise, input_blue_noise_coordinates);
	radix_threadblock_sort<SSBNPermutationBlockSize * SSBNPermutationBlockSize>(input_pixel_luminance, input_pixel_luminance_coordinates);

	short2_t luminance_coords		  = input_pixel_luminance_coordinates[thread_index_in_block];
	int luminance_global_sorted_index = luminance_coords.x + luminance_coords.y * padded_resolution_x;

	int seed_fetch_index = luminance_global_sorted_index;

	int index_x = luminance_global_sorted_index % padded_resolution_x;
	int index_y = luminance_global_sorted_index / padded_resolution_x;
	if ((index_x >= resolution_x || index_y >= resolution_y) && render_data.render_settings.sample_number == 0)
	{
		// If we're trying to fetch a seed that is in the padded area and this is the very first sample of the render, wedon't have seeds generated in the
		// padded area yet (because the renderer doesn't render the padded area = doesn't produce seeds in there). So we're just faking seeds in the padded area
		// by mirroring the seeds that are at the edge of the image
		int mirrored_x	   = index_x >= resolution_x ? (resolution_x - 1 - (index_x - resolution_x)) : index_x;
		int mirrored_y	   = index_y >= resolution_y ? (resolution_y - 1 - (index_y - resolution_y)) : index_y;
		int mirrored_index = mirrored_x + mirrored_y * padded_resolution_x;

		seed_fetch_index = mirrored_index;
	}

	int blue_noise_block_sorted_index = input_blue_noise_coordinates[thread_index_in_block];
	if (luminance_coords.x != -1 && blue_noise_block_sorted_index != -1)
		sorted_seeds[blue_noise_block_sorted_index] = in_seeds_to_sort[seed_fetch_index];

	__syncthreads();

	out_sorted_seeds_buffer[pixel_x + pixel_y * padded_resolution_x] = sorted_seeds[thread_index_in_block];
}

#endif
