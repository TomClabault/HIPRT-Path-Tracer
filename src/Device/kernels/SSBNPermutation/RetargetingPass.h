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

GLOBAL_KERNEL_SIGNATURE(void)
SSBNPermutationRetargetingPass(HIPRTRenderData render_data,
							   const int* __restrict__ blue_noise_retargeting_texture_buffer,
							   unsigned int blue_noise_texture_width,
							   unsigned int blue_noise_texture_height,
							   const unsigned int* __restrict__ sorted_seeds_buffer,
							   unsigned int* __restrict__ out_retargeted_seeds_buffer)
{
	int resolution_x = render_data.render_settings.render_resolution.x;
	int resolution_y = render_data.render_settings.render_resolution.y;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= resolution_x || y >= resolution_y)
		return;

	int blue_noise_offset_x = 0, blue_noise_offset_y = 0;
	if (render_data.render_settings.sample_number > 0)
		get_blue_noise_texture_offset(blue_noise_texture_width, blue_noise_texture_height, render_data.render_settings.sample_number, blue_noise_offset_x,
									  blue_noise_offset_y);

	int tile_base_x				= (x / blue_noise_texture_width) * blue_noise_texture_width;
	int tile_base_y				= (y / blue_noise_texture_height) * blue_noise_texture_height;
	int local_x					= x % blue_noise_texture_width;
	int local_y					= y % blue_noise_texture_height;
	int local_offset_x			= (local_x + blue_noise_offset_x) % blue_noise_texture_width;
	int local_offset_y			= (local_y + blue_noise_offset_y) % blue_noise_texture_height;
	int permutation_index_fetch = local_offset_x + local_offset_y * blue_noise_texture_width;

	int retargeted_index = blue_noise_retargeting_texture_buffer[permutation_index_fetch];

	int local_retargeted_x = retargeted_index % blue_noise_texture_width;
	int local_retargeted_y = retargeted_index / blue_noise_texture_width;

	int global_retargeted_x = tile_base_x + local_retargeted_x;
	int global_retargeted_y = tile_base_y + local_retargeted_y;

	if (global_retargeted_x >= resolution_x || global_retargeted_y >= resolution_y)
	{
		// If the retargeted pixel falls outside the screen bounds,
		// just keep it in place to maintain the bijection for the edge pixels.
		global_retargeted_x = x;
		global_retargeted_y = y;
	}

	out_retargeted_seeds_buffer[global_retargeted_x + global_retargeted_y * resolution_x] = sorted_seeds_buffer[x + y * resolution_x];
}

#endif
