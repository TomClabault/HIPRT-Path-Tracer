/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LTCS_H
#define DEVICE_INCLUDES_LTCS_H

#include "Device/includes/LightSampling/LTCs/LTCLobe.h"
#include "Device/includes/LightSampling/LTCs/LTCReadParams.h"
#include "Device/includes/Texture.h"

HIPRT_DEVICE float3 ltc_transform_cosine_to_shading(const HIPRTRenderData& render_data, float cos_theta_v, float3 direction_or_position, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_params, cos_theta_v, material, ltc_lobe);

	// Transform with the LTC: LTCMatrix * direction_or_position
	return make_float3(
		ltc_params.r * direction_or_position.x + ltc_params.g * direction_or_position.z,
		ltc_params.b * direction_or_position.y,
		// Assumes LTC[2][2]is 1.0f here
		ltc_params.a * direction_or_position.x + 1.0f * direction_or_position.z);
}

HIPRT_DEVICE float3 ltc_transform_shading_to_cosine(float3x3 matrix_inverse, float3 direction_or_position)
{
	return matrix_inverse * direction_or_position;
}

HIPRT_DEVICE float3x3 ltc_transform_shading_to_cosine_read_matrix(const HIPRTRenderData& render_data, float cos_theta_v, float3 direction_or_position, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_params, cos_theta_v, material, ltc_lobe);

	float3x3 ltc_matrix = float3x3(
		ltc_params.r, 0.0f, ltc_params.g,
		0.0f, ltc_params.b, 0.0f,
		ltc_params.a, 0.0f, 1.0f
	);

	return inverse(ltc_matrix);
}

HIPRT_DEVICE float3 ltc_transform_shading_to_cosine(const HIPRTRenderData& render_data, float cos_theta_v, float3 direction_or_position, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	float3x3 ltc_matrix_inv = ltc_transform_shading_to_cosine_read_matrix(render_data, cos_theta_v, direction_or_position, material, ltc_lobe);

	return ltc_transform_shading_to_cosine(ltc_matrix_inv, direction_or_position);
}

HIPRT_DEVICE float ltc_jacobian(const HIPRTRenderData& render_data, float cos_theta_v, float3 sampled_direction_shading_space, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_params, cos_theta_v, material, ltc_lobe);

	// TODO can we precompute that in a texture?
	float3x3 ltc_matrix_inverse = inverse(float3x3(
		ltc_params.r,	0.0f,			ltc_params.g,
		0.0f,			ltc_params.b,	0.0f,
		ltc_params.a,	0.0f,			1.0f
	));

	float3 direction_cosine_space = ltc_matrix_inverse * sampled_direction_shading_space;

	return hippt::abs(determinant(ltc_matrix_inverse)) / hippt::pow_3(hippt::length(direction_cosine_space));
}

#endif
