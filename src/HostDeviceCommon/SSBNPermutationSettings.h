/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
	bool use_world_space_hash_grid	= false;

	uint3_t* screen_space_hash_grid = nullptr;
};

#endif
