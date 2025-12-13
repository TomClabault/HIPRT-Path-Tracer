/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LTCS_H
#define DEVICE_INCLUDES_LTCS_H

#include "Device/includes/Texture.h"

enum LTCLobe
{
	COAT_LOBE = 0,
	METALLIC_LOBE = 1,
	SPECULAR_LOBE = 2,
	DIFFUSE_LOBE = 3,
};

/**
 * Given a potentially multi-layered material, this function samples one of its lobe
 * and returns which lobe was sampled along with its PDF.
 * 
 * This is meant for area-light-LTC sampling where sampling all lobes per each shading
 * point would require multiple shadow rays so instead we sample only one lobe
 * stochastically, essentially a one-sample-estimator.
 */
HIPRT_DEVICE LTCLobe ltc_lobe_sample(const DeviceUnpackedEffectiveMaterial& material, Xorshift32Generator& rng, float& out_pdf)
{
#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
	out_pdf = 1.0f;

	return LTCLobe::DIFFUSE_LOBE;
#endif

	float coat_weight = material.coat;
	float metallic_weight = material.metallic;
	float specular_weight = (1.0f - material.metallic) * material.specular;
	float diffuse_weight = specular_weight; // Same weight for diffuse and specular lobes for now, better weighting based on fresnel
	if (coat_weight + metallic_weight + specular_weight + diffuse_weight == 0.0f)
		// All lobes have 0 weight, this is the perfect only-diffuse-lobe case
		diffuse_weight = 1.0f;

	float proba_normalize = 1.0f / (coat_weight + metallic_weight + specular_weight + diffuse_weight);
	float coat_proba = coat_weight * proba_normalize;
	float metallic_proba = metallic_weight * proba_normalize;
	float specular_proba = specular_weight * proba_normalize;
	float diffuse_proba = diffuse_weight * proba_normalize;
	/*float metallic = material.metallic;
	out_metal_1_weight = metallic * outside_object;
	out_metal_2_weight = metallic * outside_object;

	float second_roughness_weight = material.second_roughness_weight;
	out_metal_1_weight = hippt::lerp(out_metal_1_weight, 0.0f, second_roughness_weight);
	out_metal_2_weight = hippt::lerp(0.0f, out_metal_2_weight, second_roughness_weight);*/

	float cdf[3];
	cdf[0] = coat_proba;
	cdf[1] = cdf[0] + metallic_proba;
	cdf[2] = cdf[1] + specular_proba;

	float random_number = rng();

	if (random_number < cdf[0])
	{
		// Coat lobe
		out_pdf = coat_proba;

		return LTCLobe::COAT_LOBE;
	}
	else if (random_number < cdf[1])
	{
		// Metallic lobe
		out_pdf = metallic_proba;

		return LTCLobe::METALLIC_LOBE;
	}
	else if (random_number < cdf[2])
	{
		// Specular lobe
		out_pdf = specular_proba;
		return LTCLobe::SPECULAR_LOBE;
	}
	else
	{
		// Diffuse lobe
		out_pdf = 1.0f - cdf[2];

		return LTCLobe::DIFFUSE_LOBE;
	}
}

HIPRT_DEVICE float ltc_lobe_eval_pdf(const DeviceUnpackedEffectiveMaterial& material, LTCLobe lobe)
{
#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
	if (lobe == LTCLobe::DIFFUSE_LOBE)
		return 1.0f;
	else
		return 0.0f;
#endif

	float coat_weight = material.coat;
	float metallic_weight = material.metallic;
	float specular_weight = (1.0f - material.metallic) * material.specular;
	float diffuse_weight = specular_weight; // Same weight for diffuse and specular lobes for now, better weighting based on fresnel
	if (coat_weight + metallic_weight + specular_weight + diffuse_weight == 0.0f)
		// All lobes have 0 weight, this is the perfect only-diffuse-lobe case
		diffuse_weight = 1.0f;

	float proba_normalize = 1.0f / (coat_weight + metallic_weight + specular_weight + diffuse_weight);

	float coat_proba = coat_weight * proba_normalize;
	float metallic_proba = metallic_weight * proba_normalize;
	float specular_proba = specular_weight * proba_normalize;
	float diffuse_proba = diffuse_weight * proba_normalize;

	switch (lobe)
	{
	case COAT_LOBE:
		return coat_proba;

	case METALLIC_LOBE:
		return metallic_proba;

	case DIFFUSE_LOBE:
		return diffuse_proba;

	case SPECULAR_LOBE:
		return specular_proba;

	default:
		return 0.0f;
	}
}

/**
 * This function returns the LTC parameters for the given cos_theta_v and material.
 * The coefficients M00, M02, M11 and M20 are stored in the R, G, B and A
 * components of the returned ColorRGBA32F respectively.
 * 
 * This function takes a random number generator and outputs a PDF because
 * the material be consisting of multiple lobes, in which case
 * this function will return the LTC matrix of one of the lobes randomly selected.
 * 
 * The PDF should be taken into account in the sampling routine using these LTC parameters.
 * The PDF parameter can be passed nullptr if not needed
 */
HIPRT_DEVICE ColorRGBA32F read_ltc_params(void* ltcs_data_param_pointer, float cos_theta_v, 
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	float roughness = 0.0f;

	switch (ltc_lobe)
	{
	case DIFFUSE_LOBE:
		return ColorRGBA32F(1.0f, 0.0f, 1.0f, 0.0f); // Identity matrix for the diffuse lobe
		break;

	case METALLIC_LOBE:
	case SPECULAR_LOBE:
		roughness = material.roughness;
		break;

	case COAT_LOBE:
		roughness = material.coat_roughness;
		break;
	}

	const void* texture_ptr = nullptr;
#ifdef __KERNELCC__
	texture_ptr = &ltcs_data_param_pointer;
#else
	texture_ptr = ltcs_data_param_pointer;
#endif

#ifdef __KERNELCC__
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, roughness * roughness);
#else
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, 1.0f - roughness * roughness);
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

HIPRT_DEVICE float3 ltc_transform_cosine_to_shading(const HIPRTRenderData& render_data, float cos_theta_v, float3 direction_or_position, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params, cos_theta_v, material, ltc_lobe);

	// Transform with the LTC: LTCMatrix * direction_or_position
	return make_float3(
		ltc_params.r * direction_or_position.x + ltc_params.g * direction_or_position.z,
		ltc_params.b * direction_or_position.y,
		// Assumes LTC[2][2]is 1.0f here
		ltc_params.a * direction_or_position.x + 1.0f * direction_or_position.z);
}

HIPRT_DEVICE float3 ltc_transform_shading_to_cosine(const HIPRTRenderData& render_data, float cos_theta_v, float3 direction_or_position, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params, cos_theta_v, material, ltc_lobe);

	float3x3 ltc_matrix = float3x3(
		ltc_params.r, 0.0f, ltc_params.g,
		0.0f, ltc_params.b, 0.0f,
		ltc_params.a, 0.0f, 1.0f
	);

	float3x3 ltc_matrix_inv = inverse(ltc_matrix);

	return ltc_matrix_inv * direction_or_position;
}

HIPRT_DEVICE float ltc_jacobian(const HIPRTRenderData& render_data, float cos_theta_v, float3 sampled_direction_shading_space, const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_specular_lambert_diffuse_ltc_params, cos_theta_v, material, ltc_lobe);

	float3x3 ltc_matrix_inverse = inverse(float3x3(
		ltc_params.r,	0.0f,			ltc_params.g,
		0.0f,			ltc_params.b,	0.0f,
		ltc_params.a,	0.0f,			1.0f
	));

	float3 direction_cosine_space = ltc_matrix_inverse * sampled_direction_shading_space;

	return hippt::abs(determinant(ltc_matrix_inverse)) / hippt::pow_3(hippt::length(direction_cosine_space));
}

#endif
