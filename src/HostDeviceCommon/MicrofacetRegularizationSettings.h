/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MICROFACET_REGULARIZATION_SETTINGS_H
#define HOST_DEVICE_COMMON_MICROFACET_REGULARIZATION_SETTINGS_H

struct MicrofacetRegularizationSettings
{
	// Maximum value that the microfacet distribution is allowed to take
	// The regularized roughness will be derived from this value
	float tau_0 = 30.0f;

	// Minimum roughness. Useful when lights are so small that even camera ray jittering
	// causes variance
	float min_roughness = 0.0f;
};

#endif
