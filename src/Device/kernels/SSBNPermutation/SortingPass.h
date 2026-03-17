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
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Pixel: (%d, %d)\n", x, y);
		printf("Thread index in block: %d\n", thread_index_in_block);
	}
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Blue noise offset: (%d, %d)\n", blue_noise_offset_x, blue_noise_offset_y);

	int blue_noise_index_x = (x + blue_noise_offset_x) % blue_noise_texture_width;
	int blue_noise_index_y = (y + blue_noise_offset_y) % blue_noise_texture_height;

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Local offset (fetching at) in blue noise texture: (%d, %d)\n", blue_noise_index_x, blue_noise_index_y);

	input_blue_noise[thread_index_in_block]				= blue_noise_dither_texture_buffer[blue_noise_index_x + blue_noise_index_y * blue_noise_texture_width];
	input_blue_noise_coordinates[thread_index_in_block] = make_short2(in_block_x, in_block_y);
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Blue noise value fetched: %d\n", input_blue_noise[thread_index_in_block]);
		printf("Blue noise coordinates in block: (%d, %d)\n", input_blue_noise_coordinates[thread_index_in_block].x,
			   input_blue_noise_coordinates[thread_index_in_block].y);
	}

	// First sample and our thread is in the padded area of the seed buffer, reading from mirrored values for the luminance
	// For pixels that are in-screen, the mirrored_pixel_index is just the pixel index
	int mirrored_at_edge_x	 = x >= resolution_x ? (resolution_x - 1 - (x - resolution_x)) : x;
	int mirrored_at_edge_y	 = y >= resolution_y ? (resolution_y - 1 - (y - resolution_y)) : y;
	int mirrored_pixel_index = mirrored_at_edge_x + mirrored_at_edge_y * resolution_x;

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Mirrored at edge coordinates: (%d, %d)\n", mirrored_at_edge_x, mirrored_at_edge_y);
		printf("Mirrored pixel index for luminance fetch: %d\n", mirrored_pixel_index);
	}

	if (!thread_valid)
	{
		// For out of frame threads
		if (render_data.render_settings.sample_number == 0)
			// For the first sample, we don't have luminance information in the padded area because the renderer doesn't render that so we're using the mirrored
			// pixel index to mirror the luminance from the actually rendered area
			input_pixel_luminance[thread_index_in_block] = render_data.buffers.last_frame_ray_colors[mirrored_pixel_index].luminance();
		else
		{
			// At higher samples, we don't want to use the mirrored luminance anymore. What we want is use the output of the sorting pass last frame, we want to
			// get as close as possible to what luminance the renderer would have produced with the sorted seeds of last frame. We don't have the exact answer
			// to that because we're not rendering padded-area pixels so we need to find an approxmiation. And using mirroring again here is certainly not going
			// to work: the sorted seeds in the padded area are not at all a mirror of the sorted seeds at the edge of the visible area so if we use mirroring,
			// we're going to get luminance which is completely uncorrelated from the seeds of the padded area = bad = white noise so we need something else

			// float approx_luminance = render_data.buffers.last_frame_ray_colors[x + y * padded_resolution_x].luminance(); // WE NEED AN APPROXMATION HERE
			float approx_luminance						 = in_seeds_to_sort[x + y * padded_resolution_x] / (float)((unsigned int)(-1));
			input_pixel_luminance[thread_index_in_block] = approx_luminance;
		}
	}
	else
		input_pixel_luminance[thread_index_in_block] = render_data.buffers.last_frame_ray_colors[mirrored_pixel_index].luminance();
	input_pixel_luminance_coordinates[thread_index_in_block] = make_short2(in_block_x, in_block_y);

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Pixel luminance fetched: %f\n", (float)input_pixel_luminance[thread_index_in_block]);
	}
	__syncthreads();

	if (in_block_x == 0 && in_block_y == 0)
	{
		bubble_sort(input_blue_noise, input_blue_noise_coordinates);
		bubble_sort(input_pixel_luminance, input_pixel_luminance_coordinates);
	}

	__syncthreads();

	int blue_noise_block_sorted_index = input_blue_noise_coordinates[thread_index_in_block].x +
										input_blue_noise_coordinates[thread_index_in_block].y * SSBNPermutationBlockSize;

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Blue noise block sorted index: %d\n", blue_noise_block_sorted_index);
	}

	unsigned int block_start_x			   = blockIdx.x * SSBNPermutationBlockSize;
	unsigned int block_start_y			   = blockIdx.y * SSBNPermutationBlockSize;
	unsigned int block_start_global_offset = block_start_x + block_start_y * padded_resolution_x;

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Block start (x, y): (%d, %d)\n", block_start_x, block_start_y);
		printf("Block start global offset: %d\n", block_start_global_offset);
	}

	short2_t luminance_coords		  = input_pixel_luminance_coordinates[thread_index_in_block];
	int luminance_global_sorted_index = block_start_global_offset + input_pixel_luminance_coordinates[thread_index_in_block].x +
										input_pixel_luminance_coordinates[thread_index_in_block].y * padded_resolution_x;

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Luminance coordinates in block: (%d, %d)\n", luminance_coords.x, luminance_coords.y);
		printf("Luminance global sorted index: %d\n", luminance_global_sorted_index);
	}

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

	sorted_seeds[blue_noise_block_sorted_index] = in_seeds_to_sort[seed_fetch_index];

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("sorted_seeds[blue_noise_block_sorted_index] = in_seeds_to_sort[luminance_global_sorted_index]; // sorted_seeds[%d] = in_seeds_to_sort[%d] = "
			   "%u\n",
			   blue_noise_block_sorted_index, luminance_global_sorted_index, sorted_seeds[blue_noise_block_sorted_index]);
	}

	__syncthreads();

	// Full pixel index that can go in the padded area of the seeds buffer
	int full_pixel_index = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * padded_resolution_x;

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Full pixel index for writing sorted seeds: %d\n", full_pixel_index);
		printf("Writing sorted seed: out_sorted_seeds_buffer[%d] = sorted_seeds[%d] = %u\n", full_pixel_index, thread_index_in_block,
			   sorted_seeds[thread_index_in_block]);
	}

	out_sorted_seeds_buffer[full_pixel_index] = sorted_seeds[thread_index_in_block];

	// if (!thread_valid)
	//	// TODO this we want the luminance associated with the seed or something
	//	render_data.buffers.last_frame_ray_colors[x + y * padded_resolution_x] = ColorRGB32F();
}

#endif
