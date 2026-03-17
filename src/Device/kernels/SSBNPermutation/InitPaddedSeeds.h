/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_SSBN_PERMUTATION_INIT_PADDED_SEEDS_H
#define KERNELS_SSBN_PERMUTATION_INIT_PADDED_SEEDS_H

#include "Device/includes/FixIntellisense.h"

GLOBAL_KERNEL_SIGNATURE(void)
SSBNPermutationInitPaddedSeeds(int resolution_x, int resolution_y, int padded_resolution_x, unsigned int* __restrict__ in_seeds_to_sort)
{
	return;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	bool thread_valid = x < resolution_x && y < resolution_y;

	if (!thread_valid)
	{
		// Thread is in the padded area of the seeds buffer, we need to fill in the padded area with "proper" seeds mirrored at the edge of the valid area
		int mirrored_at_edge_x	 = x >= resolution_x ? (resolution_x - 1 - (x - resolution_x)) : x;
		int mirrored_at_edge_y	 = y >= resolution_y ? (resolution_y - 1 - (y - resolution_y)) : y;
		int mirrored_pixel_index = mirrored_at_edge_x + mirrored_at_edge_y * padded_resolution_x;

		int full_pixel_index			   = x + y * padded_resolution_x;
		in_seeds_to_sort[full_pixel_index] = in_seeds_to_sort[mirrored_pixel_index];
	}
}

#endif
