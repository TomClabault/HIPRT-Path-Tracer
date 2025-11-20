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

HIPRT_DEVICE float3 ltc_transform_cosine_to_shading(const HIPRTRenderData& render_data, float cos_theta_v, float roughness, float3 direction_or_position)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params, cos_theta_v, roughness);

	// Transform with the LTC: LTCMatrix * direction_or_position
	return make_float3(
		ltc_params.r * direction_or_position.x + ltc_params.g * direction_or_position.z,
		ltc_params.b * direction_or_position.y,
		// Assumes LTC[2][2]is 1.0f here
		ltc_params.a * direction_or_position.x + 1.0f * direction_or_position.z);
}

HIPRT_DEVICE float3 ltc_transform_shading_to_cosine(const HIPRTRenderData& render_data, float cos_theta_v, float roughness, float3 direction_or_position)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params, cos_theta_v, roughness);

	float3x3 ltc_matrix = float3x3(
		ltc_params.r, 0.0f, ltc_params.g,
		0.0f, ltc_params.b, 0.0f,
		ltc_params.a, 0.0f, 1.0f
	);

	float3x3 ltc_matrix_inv = inverse(ltc_matrix);

	return ltc_matrix_inv * direction_or_position;
	// Transform with the LTC: LTCMatrix * direction_or_position
	//return make_float3(
	//	ltc_params.r * direction_or_position.x + ltc_params.g * direction_or_position.z,
	//	ltc_params.b * direction_or_position.y,
	//	// Assumes LTC[2][2]is 1.0f here
	//	ltc_params.a * direction_or_position.x + 1.0f * direction_or_position.z);
}

HIPRT_DEVICE float ltc_jacobian(const HIPRTRenderData& render_data, float cos_theta_v, float roughness, float3 sampled_direction_shading_space)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params, cos_theta_v, roughness);

	float3x3 ltc_matrix_inverse = inverse(float3x3(
		ltc_params.r,	0.0f,			ltc_params.g,
		0.0f,			ltc_params.b,	0.0f,
		ltc_params.a,	0.0f,			1.0f
	));

	float3 direction_cosine_space = ltc_matrix_inverse * sampled_direction_shading_space;
	return hippt::abs(determinant(ltc_matrix_inverse)) / hippt::pow_3(hippt::length(direction_cosine_space));
}

#endif
