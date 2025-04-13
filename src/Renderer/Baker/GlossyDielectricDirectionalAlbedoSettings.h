/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DIELECTRIC_FRESNEL_DIRECTIONAL_ALBEDO_SETTINGS_H
#define DIELECTRIC_FRESNEL_DIRECTIONAL_ALBEDO_SETTINGS_H

#include "Renderer/Baker/GPUBakerConstants.h"

struct GlossyDielectricDirectionalAlbedoSettings
{
	int texture_size_cos_theta_o = GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O;
	int texture_size_roughness = GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS;
	int texture_size_ior = GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR;
	GGXMaskingShadowingFlavor masking_shadowing_term;

	int integration_sample_count = 131072;
};

#endif
