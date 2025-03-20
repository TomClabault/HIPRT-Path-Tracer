/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MICROFACET_REGULARIZATION_SETTINGS_H
#define HOST_DEVICE_COMMON_MICROFACET_REGULARIZATION_SETTINGS_H

struct MicrofacetRegularizationSettings
{
	bool DEBUG_DO_REGULARIZATION = false;

	// Maximum value that the microfacet distribution is allowed to take
	// The regularized roughness will be derived from this value
	float tau = 100.0f;
};

#endif
