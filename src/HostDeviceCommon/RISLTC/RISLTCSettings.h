/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RISLTC_SETTINGS_H
#define HOST_DEVICE_COMMON_RISLTC_SETTINGS_H

struct RISLTCSettings
{
	// How many candidate lights to sample for RISLTC (Shash et. al, 2023)
	int number_of_light_candidates = 4;
	// How many candidates samples from the BSDF to use in combination
	// with the light candidates for RISLTC
	int number_of_bsdf_candidates = 0;
};

#endif
