/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RIS_SETTINGS_H
#define HOST_DEVICE_COMMON_RIS_SETTINGS_H

struct RISSettings
{
	// How many candidate lights to sample for RIS (Resampled Importance Sampling)
	int number_of_light_candidates = 4;
	// How many candidates samples from the BSDF to use in combination
	// with the light candidates for RIS
	int number_of_bsdf_candidates = 1;
};

#endif
