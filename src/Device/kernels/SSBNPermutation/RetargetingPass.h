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
	int resolution_x		= render_data.render_settings.render_resolution.x;
	int resolution_y		= render_data.render_settings.render_resolution.y;
	int padded_resolution_x = (resolution_x + blue_noise_texture_width - 1) / blue_noise_texture_width * blue_noise_texture_width;
	int padded_resolution_y = (resolution_y + blue_noise_texture_height - 1) / blue_noise_texture_height * blue_noise_texture_height;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= padded_resolution_x || y >= padded_resolution_y)
		// This should never happen because the launch dimensions should be exactly the padded resolution, but just in case
		return;

	int blue_noise_offset_x = 0, blue_noise_offset_y = 0;
	get_blue_noise_texture_offset(blue_noise_texture_width, blue_noise_texture_height, render_data.render_settings.sample_number, blue_noise_offset_x,
								  blue_noise_offset_y);

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Pixel: (%d, %d)\n", x, y);
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Blue noise offset: (%d, %d)\n", blue_noise_offset_x, blue_noise_offset_y);

	int local_offset_x = (x + blue_noise_offset_x) % blue_noise_texture_width;
	int local_offset_y = (y + blue_noise_offset_y) % blue_noise_texture_height;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Local offset (fetching at) in blue noise texture: (%d, %d)\n", local_offset_x, local_offset_y);

	int permutation_index_fetch = local_offset_x + local_offset_y * blue_noise_texture_width;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Permutation index fetch: %d\n", permutation_index_fetch);

	int local_retargeted_index = blue_noise_retargeting_texture_buffer[permutation_index_fetch];
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("Local retargeted index: %d\n", local_retargeted_index);

	int local_retargeted_x = local_retargeted_index % blue_noise_texture_width;
	int local_retargeted_y = local_retargeted_index / blue_noise_texture_width;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("int local_retargeted_x = local_retargeted_index %% blue_noise_texture_width; // %d\n", local_retargeted_x);
		printf("int local_retargeted_y = local_retargeted_index / blue_noise_texture_width; // %d\n", local_retargeted_y);
	}

	local_retargeted_x -= blue_noise_offset_x;
	local_retargeted_y -= blue_noise_offset_y;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("local_retargeted_x -= blue_noise_offset_x; // %d\n", local_retargeted_x);
		printf("local_retargeted_y -= blue_noise_offset_y; // %d\n", local_retargeted_y);
	}

	local_retargeted_x %= blue_noise_texture_width;
	local_retargeted_y %= blue_noise_texture_height;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("local_retargeted_x %%= blue_noise_texture_width; // %d\n", local_retargeted_x);
		printf("local_retargeted_y %%= blue_noise_texture_height; // %d\n", local_retargeted_y);
	}
	// Modulo wrapping accounting for potentially negative local_retargeted_*
	local_retargeted_x = (local_retargeted_x + blue_noise_texture_width) % blue_noise_texture_width;
	local_retargeted_y = (local_retargeted_y + blue_noise_texture_height) % blue_noise_texture_height;
	/*local_retargeted_x = (local_retargeted_x % blue_noise_texture_width + blue_noise_texture_width) % blue_noise_texture_width;
	local_retargeted_y = (local_retargeted_y % blue_noise_texture_height + blue_noise_texture_height) % blue_noise_texture_height;*/
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("After modulo wrapping:\n");
		printf("local_retargeted_x = %d\n", local_retargeted_x);
		printf("local_retargeted_y = %d\n", local_retargeted_y);
	}

	int tile_base_x = (x / blue_noise_texture_width) * blue_noise_texture_width;
	int tile_base_y = (y / blue_noise_texture_height) * blue_noise_texture_height;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Tile base:\n");
		printf("tile_base_x = (x / blue_noise_texture_width) * blue_noise_texture_width; // %d\n", tile_base_x);
		printf("tile_base_y = (y / blue_noise_texture_height) * blue_noise_texture_height; // %d\n", tile_base_y);
	}

	int global_retargeted_x = tile_base_x + local_retargeted_x;
	int global_retargeted_y = tile_base_y + local_retargeted_y;
	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
	{
		printf("Global retargeted pixel:\n");
		printf("global_retargeted_x = tile_base_x + local_retargeted_x; // %d\n", global_retargeted_x);
		printf("global_retargeted_y = tile_base_y + local_retargeted_y; // %d\n", global_retargeted_y);
	}

	if (global_retargeted_x >= padded_resolution_x || global_retargeted_y >= padded_resolution_y)
	{
		// This should never happen because the seed buffer is padded to the next multiple of the blue noise texture dimensions, which means that we should
		// never get out of bounds

		// If the retargeted pixel falls outside the screen bounds,
		// just keep it in place to maintain the bijection for the edge pixels.
		global_retargeted_x = x;
		global_retargeted_y = y;

		static bool done = false;
		if (!done)
		{
			printf("WHAT?\n");

			done = true;
		}
	}

	if (x == resolution_x - 1 && y == 18 && render_data.render_settings.sample_number == 1)
		printf("\n\n\n");

	out_retargeted_seeds_buffer[x + y * padded_resolution_x] = sorted_seeds_buffer[global_retargeted_x + global_retargeted_y * padded_resolution_x];
}

#endif
