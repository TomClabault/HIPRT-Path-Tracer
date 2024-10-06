/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_WORLD_SETTINGS_H
#define HOST_DEVICE_COMMON_WORLD_SETTINGS_H

#include "HostDeviceCommon/Color.h"

enum AmbientLightType
{
	NONE,
	UNIFORM,
	ENVMAP
};

struct WorldSettings
{
	AmbientLightType ambient_light_type = AmbientLightType::NONE;
	ColorRGB32F uniform_light_color = ColorRGB32F(0.5f);

	// Width and height in pixels. Both in the range [1, XXX]
	unsigned int envmap_width = 0, envmap_height = 0;
	// Simple scale multiplier on the envmap color read from the envmap texture
	// in the shader
	float envmap_intensity = 1.0f;
	// If true, the background of the scene (where rays directly miss any geometry
	// and we directly see the skysphere) will scale with the envmap_intensity coefficient.
	// This can be visually unpleasing because the background will most likely
	// become completely white and blown out.
	int envmap_scale_background_intensity = false;
	// This void pointer is a either a float* for the CPU
	// or a oroTextureObject_t for the GPU.
	// Proper reinterpreting of the pointer is done in the kernel.
	void* envmap = nullptr;

	// Luminance sum of all the texels of the envmap
	float envmap_total_sum = 0.0f;

	// Cumulative distribution function. 1D float array of length width * height for
	// importance sampling the envmap with a binary search strategy
	float* envmap_cdf = nullptr;

	// Probabilities and aliases for sampling the envmap with the alias table strategy
	int* alias_table_alias = nullptr;
	float* alias_table_probas = nullptr;

	// Rotation matrix for rotating the envmap around in the current frame
	float4x4 envmap_to_world_matrix = float4x4{ { {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f } } };
	float4x4 world_to_envmap_matrix = float4x4{ { {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f } } };
};

#endif
