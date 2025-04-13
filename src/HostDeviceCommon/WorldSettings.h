/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_WORLD_SETTINGS_H
#define HOST_DEVICE_COMMON_WORLD_SETTINGS_H

#include "Device/includes/AliasTable.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Packing.h"

enum AmbientLightType
{
	NONE,
	UNIFORM,
	ENVMAP
};

struct WorldSettings
{
	AmbientLightType ambient_light_type = AmbientLightType::UNIFORM;
	ColorRGB32F uniform_light_color = ColorRGB32F(5.0f);

	// Width and height in pixels. Both in the range [1, XXX]
	unsigned int envmap_width = 0, envmap_height = 0;
	// Simple scale multiplier on the envmap color read from the envmap texture
	// in the shader
	float envmap_intensity = 5.0f;
	// If true, the background of the scene (where rays directly miss any geometry
	// and we directly see the skysphere) will scale with the envmap_intensity coefficient.
	// This can be visually unpleasing because the background will most likely
	// become completely white and blown out.
	int envmap_scale_background_intensity = false;

	// Packed RGBE 9/9/9/5 envmap texels
	RGBE9995Packed* envmap;

	// Luminance sum of all the texels of the envmap
	float envmap_total_sum = 0.0f;

	// Cumulative distribution function. 1D float array of length width * height for
	// importance sampling the envmap with a binary search strategy
	float* envmap_cdf = nullptr;

	// Probabilities and aliases for sampling the envmap with the alias table strategy
	DeviceAliasTable envmap_alias_table;

	// Rotation matrix for rotating the envmap around in the current frame
	float3x3 envmap_to_world_matrix = float3x3{ { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} } };
	float3x3 world_to_envmap_matrix = float3x3{ { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} } };
};

#endif
