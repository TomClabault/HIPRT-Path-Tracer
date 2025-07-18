/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef NEE_PLUS_PLUS_CACHING_PREPASS_KERNEL_PARAMETERS_H
#define NEE_PLUS_PLUS_CACHING_PREPASS_KERNEL_PARAMETERS_H

#include "Device/includes/NEE++/NEE++.h"

struct NEEPlusPlusCachingPrepassParameters
{
	NEEPlusPlusDevice nee_plus_plus;

	HIPRTCamera current_camera;

	unsigned int random_number = 42;
};

#endif