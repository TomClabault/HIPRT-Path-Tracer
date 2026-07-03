/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_SSBN_PERMUTATION_SETTINGS_H
#define HOST_DEVICE_SSBN_PERMUTATION_SETTINGS_H

#include "HostDeviceCommon/Maths/VecTypes.h"

struct SSBNPermutationSettings
{
	unsigned int blue_noise_texture_width  = 128;
	unsigned int blue_noise_texture_height = 128;

	bool use_screen_space_hash_grid = true;
	bool use_surface_normal			= true;

	// This is used to force the random seeds to not be reset when we reset the render. Useful to accumulate blue noise quality with SSBN permutations,
	// otherwise SSBN permutation always needs more than 1SPP to kick in. This breaks determinism though as all 1SPP frame will be different!
	bool accumulate_blue_noise_1spp = true;

	uint3_t* screen_space_hash_grid = nullptr;
};

#endif
