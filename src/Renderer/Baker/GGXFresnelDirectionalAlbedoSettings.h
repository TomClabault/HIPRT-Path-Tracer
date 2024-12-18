/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GGX_FRESNEL_DIRECTIONAL_ALBEDO_SETTINGS_H
#define GGX_FRESNEL_DIRECTIONAL_ALBEDO_SETTINGS_H

#include "Renderer/Baker/GPUBakerConstants.h"

struct GGXFresnelDirectionalAlbedoSettings
{
	int texture_size_cos_theta = GPUBakerConstants::GGX_FRESNEL_ESS_TEXTURE_SIZE_COS_THETA_O;
	int texture_size_roughness = GPUBakerConstants::GGX_FRESNEL_ESS_TEXTURE_SIZE_ROUGHNESS;
	int texture_size_ior = GPUBakerConstants::GGX_FRESNEL_ESS_TEXTURE_SIZE_IOR;

	int integration_sample_count = 65536;
};

#endif
