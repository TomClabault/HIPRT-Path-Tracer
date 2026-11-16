/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
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
	float3_t scene_min = make_float3(0.0f, 0.0f, 0.0f);
	float padding_;
	float3_t scene_max = make_float3(0.0f, 0.0f, 0.0f);
	float padding__;

	AmbientLightType ambient_light_type = AmbientLightType::UNIFORM;
	ColorRGB32F uniform_light_color		= ColorRGB32F(0.0f);

	// Sets the width and height of the envmap and their float copies
	HIPRT_HOST void set_envmap_dimensions(unsigned int width, unsigned int height)
	{
		envmap_width  = width;
		envmap_height = height;

		envmap_width_as_float			 = static_cast<float>(width);
		envmap_height_as_float			 = static_cast<float>(height);
		envmap_width_minus_one_as_float	 = static_cast<float>(width - 1);
		envmap_height_minus_one_as_float = static_cast<float>(height - 1);
		envmap_texel_count_as_float		 = static_cast<float>(width * height);
	}

	// Width and height in pixels. Both in the range [1, XXX]. Set with set_envmap_dimensions()
	unsigned int envmap_width = 0, envmap_height = 0;

	// Float copies of the dimensions used by the envmap sampling and lookups.
	//
	// On the GPU, converting a uniform int to a float is a vector instruction: when the compiler hoists that conversion out of a loop (the bounce
	// loop for example), the converted value keeps a vector register for the whole loop. A float read from memory is used directly from a scalar
	// register instead
	float envmap_width_as_float			   = 0.0f;
	float envmap_height_as_float		   = 0.0f;
	float envmap_width_minus_one_as_float  = 0.0f;
	float envmap_height_minus_one_as_float = 0.0f;
	// envmap_width * envmap_height
	float envmap_texel_count_as_float = 0.0f;

	// Simple scale multiplier on the envmap color read from the envmap texture
	// in the shader
	float envmap_intensity = 1.0f;
	// This value may be different from 1.0f in case the original maximum texel intensity of the envmap exceeds the maximum representable value of the RGBE
	// 9/9/9/5 format. In that case, the envmap is scaled down to fit in the RGBE 9/9/9/5 format and this scaling factor is stored here. This scaling factor is
	// applied to the envmap texels when reading them in the shader.
	float envmap_packed_scaling_factor = 1.0f;
	// If true, the background of the scene (where rays directly miss any geometry
	// and we directly see the skysphere) will scale with the envmap_intensity coefficient.
	// This can be visually unpleasing because the background will most likely
	// become completely white and blown out.
	int envmap_scale_background_intensity = false;

	// Packed RGBE 9/9/9/5 envmap texels
	RGBE9995Packed* envmap;

	// Luminance sum of all the texels of the envmap
	float envmap_total_sum = 0.0f;
	// Mean luminance of the envmap over the sphere of directions (texels weighted by their solid angle), before 'envmap_intensity'
	float envmap_solid_angle_mean_luminance = 0.0f;

	// Cumulative distribution function. 1D float array of length width * height for
	// importance sampling the envmap with a binary search strategy
	float* envmap_cdf = nullptr;

	// Probabilities and aliases for sampling the envmap with the alias table strategy
	AliasTableDevice envmap_alias_table;

	// Rotation matrix for rotating the envmap around in the current frame
	float3x3 envmap_to_world_matrix = float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);

	float3x3 world_to_envmap_matrix = float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
};

#endif // #ifndef HOST_DEVICE_COMMON_WORLD_SETTINGS_H
