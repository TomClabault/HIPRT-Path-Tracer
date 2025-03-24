/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_SETTINGS_H
#define GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_SETTINGS_H

#include "Renderer/Baker/GPUBakerConstants.h"

struct GGXThinGlassDirectionalAlbedoSettings
{
	int texture_size_cos_theta_o = GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O;
	int texture_size_roughness = GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS;
	int texture_size_ior = GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR;
	GGXMaskingShadowingFlavor masking_shadowing_term;

	int integration_sample_count = 65536;
};

#endif
