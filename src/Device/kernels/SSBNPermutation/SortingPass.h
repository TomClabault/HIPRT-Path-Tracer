/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_SSBN_PERMUTATION_SORTING_PASS_H
#define KERNELS_SSBN_PERMUTATION_SORTING_PASS_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/Compute/RadixSortBlock.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/SSBNPermutation/SSBNPermutationCommon.h"
#include "HostDeviceCommon/KernelOptions/SSBNPermutationOptions.h"
#include "HostDeviceCommon/RenderData.h"

#define INVALID_BLUE_NOISE_VALUE ((short int)32767)
#define INVALID_LUMINANCE_VALUE	 ((unsigned short int)-1)

/**
 * Sorts key/value pairs where the keys can only take values 0-255. The keys are short int to allow for encoding an INVALID_VALUE. Those invalid values will be
 * sorted at the end of the array
 */
template <typename K_v, typename V_t>
HIPRT_DEVICE void sort_counting_sort_8b_with_invalid_values(K_v* keys_uchar, V_t* values, K_v INVALID_VALUE)
{
	int thread_index_in_block = threadIdx.x;

	__shared__ int invalid_values_count;
	__shared__ int values_counts[256];

	for (int i = thread_index_in_block; i < 256; i += blockDim.x * blockDim.y)
		values_counts[i] = 0;
	invalid_values_count = 0;

	__syncthreads();

	K_v value = keys_uchar[thread_index_in_block];
	if (value != INVALID_VALUE)
		hippt::atomic_fetch_add_gpu(&values_counts[value], 1);

	__syncthreads();

	// Exclusive prefix sum of the counts
	__shared__ int warps_inclusive_scans[8];

	if constexpr (SSBNPermutationBlockSize * SSBNPermutationBlockSize < 256)
	{
		int running_warp_prefix_sum = 0;

		int warp_index	= thread_index_in_block >> 5;
		int warp_offset = 0;
		while (warp_index + warp_offset < 8)
		{
			int warp_index_offsetted = warp_index + warp_offset;

			int lane				= thread_index_in_block & 31;
			int element_index		= warp_index_offsetted * 32 + lane;
			int bn_value_count		= element_index < 256 ? values_counts[element_index] : 0;
			int warp_inclusive_scan = warp_prefix_scan_inclusive(bn_value_count);

			if (lane == 31)
				warps_inclusive_scans[warp_index_offsetted] = warp_inclusive_scan;

			__syncthreads();

			// The first thread scans the values written so far by the warps
			int warps_per_block = (blockDim.x * blockDim.y) >> 5;
			if (thread_index_in_block == 0)
			{
				for (int w = warp_offset; w < warp_offset + warps_per_block && w < 8; ++w)
				{
					int warp_inclusive = warps_inclusive_scans[w];

					warps_inclusive_scans[w] = running_warp_prefix_sum;

					running_warp_prefix_sum += warp_inclusive;
				}
			}

			__syncthreads();

			if (element_index < 256)
			{
				int warp_exclusive_scan = warp_inclusive_scan - bn_value_count;
				int warp_base			= warps_inclusive_scans[warp_index_offsetted];

				values_counts[element_index] = warp_exclusive_scan + warp_base;
			}

			warp_offset += warps_per_block;
		}
	}
	else
	{
		// We don't need a while loop for this one
		int warp_index			= thread_index_in_block >> 5;
		int lane				= thread_index_in_block & 31;
		int bn_value_count		= thread_index_in_block < 256 ? values_counts[thread_index_in_block] : 0;
		int warp_inclusive_scan = warp_prefix_scan_inclusive(bn_value_count, thread_index_in_block);

		if (lane == 31)
			warps_inclusive_scans[warp_index] = warp_inclusive_scan;

		__syncthreads();

		// Thread 0 does the scan of the 8 values
		if (thread_index_in_block == 0)
		{
			int running = 0;

			for (int w = 0; w < 8; ++w)
			{
				int warp_inclusive = warps_inclusive_scans[w];

				warps_inclusive_scans[w] = running;

				running += warp_inclusive;
			}
		}

		__syncthreads();

		if (thread_index_in_block < 256)
		{
			int warp_exclusive_scan = warp_inclusive_scan - bn_value_count;
			int warp_base			= warps_inclusive_scans[warp_index];

			values_counts[thread_index_in_block] = warp_exclusive_scan + warp_base;
		}
	}

	__syncthreads();

	int value_sorted_index;
	if (value != INVALID_VALUE)
		value_sorted_index = hippt::atomic_fetch_add_gpu(&values_counts[value], 1);
	else
	{
		int invalid_value_index = hippt::atomic_fetch_add_gpu(&invalid_values_count, 1);

		// Putting the invalid values at the end
		value_sorted_index = SSBNPermutationBlockSize * SSBNPermutationBlockSize - 1 - invalid_value_index;
	}

	__shared__ short int sorted_keys[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ V_t value_sorted_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];

	sorted_keys[value_sorted_index]				 = value;
	value_sorted_coordinates[value_sorted_index] = values[thread_index_in_block];

	__syncthreads();

	keys_uchar[thread_index_in_block] = sorted_keys[thread_index_in_block];
	values[thread_index_in_block]	  = value_sorted_coordinates[thread_index_in_block];
}

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char SSBN_PERMUTATION_RENDER_DATA[sizeof(HIPRTRenderData)];
}
GLOBAL_KERNEL_SIGNATURE(void)
SSBNPermutationSortingPass(unsigned char* __restrict__ blue_noise_dither_texture_buffer,
						   unsigned int blue_noise_texture_width,
						   unsigned int blue_noise_texture_height,
						   unsigned int* __restrict__ in_seeds_to_sort,
						   unsigned int* __restrict__ out_sorted_seeds_buffer,
						   const int* __restrict__ in_hash_grid_cell_offsets_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline SSBNPermutationSortingPass(HIPRTRenderData render_data,
								  unsigned char* __restrict__ blue_noise_dither_texture_buffer,
								  unsigned int blue_noise_texture_width,
								  unsigned int blue_noise_texture_height,
								  unsigned int* __restrict__ in_seeds_to_sort,
								  unsigned int* __restrict__ out_sorted_seeds_buffer,
								  const int* __restrict__ in_hash_grid_cell_offsets_buffer)
#endif
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(SSBN_PERMUTATION_RENDER_DATA);
#endif
	int resolution_x = render_data.render_settings.render_resolution.x;
	int resolution_y = render_data.render_settings.render_resolution.y;

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

	// TODO can we store these in 1 unsigned int array with some packing and sort them together to avoid having 2 shared mem arrays = 2x less shared mem
	// accesses?
	__shared__ short int input_blue_noise[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ short int input_blue_noise_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	// input_pixel_luminance are fp16 but with bits reinterpreted as unsigned short int for the radix sort. We don't have negative luminance or NaNs/special
	// values so reinterpreting is fine
	//
	// TODO approximate this with unsigned char and scale with the maximum value to have the maximum spread in the unsigned char and use counting sort as well?
	// TODO can we store these in 1 unsigned_int3 array with some packing and sort them together to avoid having 2 shared mem arrays = 2x less shared mem
	// accesses?
	__shared__ unsigned short int input_pixel_luminance[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ short2_t input_pixel_luminance_coordinates[SSBNPermutationBlockSize * SSBNPermutationBlockSize];
	__shared__ unsigned int sorted_seeds[SSBNPermutationBlockSize * SSBNPermutationBlockSize];

	input_pixel_luminance[thread_index_in_block] = 0;
	sorted_seeds[thread_index_in_block]			 = 0;

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
		input_blue_noise[thread_index_in_block]				= INVALID_BLUE_NOISE_VALUE;
		input_blue_noise_coordinates[thread_index_in_block] = -1;
	}

	// First sample and our thread is in the padded area of the seed buffer, reading from mirrored values for the luminance
	// For pixels that are in-screen, the mirrored_pixel_index is just the pixel index
	int mirrored_at_edge_x	 = pixel_x >= resolution_x ? (resolution_x - 1 - (pixel_x - resolution_x)) : pixel_x;
	int mirrored_at_edge_y	 = pixel_y >= resolution_y ? (resolution_y - 1 - (pixel_y - resolution_y)) : pixel_y;
	int mirrored_pixel_index = mirrored_at_edge_x + mirrored_at_edge_y * resolution_x;

	fp16 luminance = 0.0f;
	if (valid_input_data && thread_valid)
		luminance = (fp16)hippt::intrin_logf(1.0f + render_data.buffers.last_frame_ray_colors[mirrored_pixel_index].luminance());

	// All threads participate in this to find the maximum. Invalid threads have luminance 0 so they do not interfere with the maximum
	fp16 max_luminance_value = block_reduce<SSBNPermutationBlockSize * SSBNPermutationBlockSize, fp16, OperatorMax<fp16>>(luminance, thread_index_in_block);

	input_pixel_luminance[thread_index_in_block]			 = static_cast<unsigned short int>((float)luminance / (float)max_luminance_value * 255);
	input_pixel_luminance_coordinates[thread_index_in_block] = make_short2(pixel_x, pixel_y);

	if (!valid_input_data || !thread_valid)
	{
		input_pixel_luminance[thread_index_in_block]			 = INVALID_LUMINANCE_VALUE;
		input_pixel_luminance_coordinates[thread_index_in_block] = make_short2(-1, -1);
	}

	__syncthreads();

	// Simple counting sort instead of radix sort for sorting the blue noise because there are only 256 possible values
	sort_counting_sort_8b_with_invalid_values(input_blue_noise, input_blue_noise_coordinates, INVALID_BLUE_NOISE_VALUE);
	sort_counting_sort_8b_with_invalid_values(input_pixel_luminance, input_pixel_luminance_coordinates, INVALID_LUMINANCE_VALUE);

	short2_t luminance_coords		  = input_pixel_luminance_coordinates[thread_index_in_block];
	int luminance_global_sorted_index = luminance_coords.x + luminance_coords.y * resolution_x;

	int seed_fetch_index = luminance_global_sorted_index;

	int index_x = luminance_global_sorted_index % resolution_x;
	int index_y = luminance_global_sorted_index / resolution_x;
	if ((index_x >= resolution_x || index_y >= resolution_y) && render_data.render_settings.sample_number == 0)
	{
		// If we're trying to fetch a seed that is in the padded area and this is the very first sample of the render, wedon't have seeds generated in the
		// padded area yet (because the renderer doesn't render the padded area = doesn't produce seeds in there). So we're just faking seeds in the padded area
		// by mirroring the seeds that are at the edge of the image
		int mirrored_x	   = index_x >= resolution_x ? (resolution_x - 1 - (index_x - resolution_x)) : index_x;
		int mirrored_y	   = index_y >= resolution_y ? (resolution_y - 1 - (index_y - resolution_y)) : index_y;
		int mirrored_index = mirrored_x + mirrored_y * resolution_x;

		seed_fetch_index = mirrored_index;
	}

	int blue_noise_block_sorted_index = input_blue_noise_coordinates[thread_index_in_block];
	if (luminance_coords.x != -1 && blue_noise_block_sorted_index != -1)
		sorted_seeds[blue_noise_block_sorted_index] = in_seeds_to_sort[seed_fetch_index];

	__syncthreads();

	out_sorted_seeds_buffer[pixel_x + pixel_y * resolution_x] = sorted_seeds[thread_index_in_block];
}

#endif
