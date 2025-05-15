/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_SHADING_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_SHADING_SETTINGS_H

struct ReGIRShadingSettings
{
	// At path tracing time, how many reservoirs of the grid cell of the point we're trying to shade
	// are going to be resampled (with the BRDF term) to produce the final light sample used for NEE
	int cell_reservoir_resample_per_shading_point = 1;
	// Whether or not to jitter the world space position used when looking up the ReGIR grid
	// This helps eliminate grid discretization  artifacts
	bool do_cell_jittering = true;
};

#endif
