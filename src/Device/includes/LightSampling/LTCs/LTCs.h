/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LTCS_H
#define DEVICE_INCLUDES_LTCS_H

#include "Device/includes/Texture.h"

HIPRT_DEVICE ColorRGBA32F read_ltc_params(void* ltcs_data_param_pointer, float cos_theta_v, float roughness)
{
	const void* texture_ptr = nullptr;
#ifdef __KERNELCC__
	texture_ptr = &ltcs_data_param_pointer;
#else
	texture_ptr = ltcs_data_param_pointer;
#endif

#ifdef __KERNELCC__
	// We want to flip the UV-Y on the GPU but the textures are using
	// clamp addressing mode so we can't use the parameter 'flip UV-Y' of sample_texture_rgba_32bits.
	float2 uv = make_float2(roughness, acos(cos_theta_v) / hippt::M_PI_TWO);
#else
	float2 uv = make_float2(roughness, 1.0f - acos(cos_theta_v) / hippt::M_PI_TWO);
#endif

	// R, G, B and A components represent respectively:
	// m00, m02, m11 and m20 of the LTC matrix
	//
	// The matrix is thus:
	// 	{ m00  0   m02 }
	//  { 0    m11  0  }
	//  { m20  0    1  }
	ColorRGBA32F ltc_params = sample_texture_rgba_32bits(texture_ptr, 0, /* is_srgb */ false, uv, /* flip UV-Y */ false);

	return ltc_params;
}

HIPRT_DEVICE float3 ltc_transform(void* ltcs_data_param_pointer, float cos_theta_v, float roughness, float3 direction_or_position)
{
	ColorRGBA32F ltc_params = read_ltc_params(ltcs_data_param_pointer, cos_theta_v, roughness);

	float3x3 ltc_matrix = float3x3(
		ltc_params.r, 0.0f, ltc_params.g,
		0.0f, ltc_params.b, 0.0f,
		ltc_params.a, 0.0f, 1.0f
	);

	return ltc_matrix * direction_or_position;
	// printf("Params: R=%f, G=%f, B=%f, A=%f\n", ltc_params.r, ltc_params.g, ltc_params.b, ltc_params.a);
	
	// Transform with the LTC: LTCMatrix * direction_or_position
	return make_float3(
		ltc_params.r * direction_or_position.x + ltc_params.g * direction_or_position.z,
		ltc_params.b * direction_or_position.y,
		// Assumes LTC[2][2]is 1.0f here
		ltc_params.a * direction_or_position.x + 1.0f * direction_or_position.z);
}

HIPRT_DEVICE float ltc_jacobian(void* ltcs_data_param_pointer, float cos_theta_v, float roughness, float3 sampled_direction_shading_space)
{
	ColorRGBA32F ltc_params = read_ltc_params(ltcs_data_param_pointer, cos_theta_v, roughness);

	float3x3 ltc_matrix = float3x3(
		ltc_params.r,	0.0f,			ltc_params.g,
		0.0f,			ltc_params.b,	0.0f,
		ltc_params.a,	0.0f,			1.0f
	);

	return hippt::abs(determinant(ltc_matrix)) / hippt::pow_3(hippt::length(ltc_matrix * sampled_direction_shading_space));
}

#endif
