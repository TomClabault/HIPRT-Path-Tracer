/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_CONSTANTS_H
#define HOST_DEVICE_COMMON_MATERIAL_CONSTANTS_H

struct MaterialConstants
{
	static constexpr int NO_TEXTURE = 65535;
	// When an emissive texture is read and is determine to be
	// constant, no emissive texture will be used. Instead,
	// we'll just set the emission of the material to that constant emission value
	// and the emissive texture index of the material will be replaced by
	// CONSTANT_EMISSIVE_TEXTURE
	static constexpr int CONSTANT_EMISSIVE_TEXTURE = 65534;
	// Maximum number of different textures per scene
	static constexpr int MAX_TEXTURE_COUNT = 65533;

	static constexpr float ROUGHNESS_CLAMP = 1.0e-4f;
	static constexpr float PERFECTLY_SMOOTH_ROUGHNESS_THRESHOLD = 1.0e-2f;
	static constexpr float DELTA_DISTRIBUTION_HIGH_VALUE = 1.0e9f;
	static constexpr float DELTA_DISTRIBUTION_ALIGNEMENT_THRESHOLD = 0.999999f;
};

#endif
