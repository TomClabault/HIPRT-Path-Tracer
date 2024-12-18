/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GGX_CONDUCTOR_DIRECTIONAL_ALBEDO_SETTINGS_H
#define GGX_CONDUCTOR_DIRECTIONAL_ALBEDO_SETTINGS_H

#include "Renderer/Baker/GPUBakerConstants.h"

struct GGXConductorDirectionalAlbedoSettings
{
	int texture_size_cos_theta = GPUBakerConstants::GGX_CONDUCTOR_ESS_TEXTURE_SIZE_COS_THETA_O;
	int texture_size_roughness = GPUBakerConstants::GGX_CONDUCTOR_ESS_TEXTURE_SIZE_ROUGHNESS;

	int integration_sample_count = 65536;
};

#endif
