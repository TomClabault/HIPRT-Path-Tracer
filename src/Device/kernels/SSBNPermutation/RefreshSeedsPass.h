/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_SSBN_PERMUTATION_REFRESH_SEEDS_PASS_H
#define KERNELS_SSBN_PERMUTATION_REFRESH_SEEDS_PASS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Random.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) SSBNPermutationRefreshSeedsPass(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline SSBNPermutationRefreshSeedsPass(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

	render_data.store_input_random_seed(pixel_index, generate_fresh_pixel_random_seed(render_data, pixel_index));
}

#endif
