/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTC_LOBE_UTILS_H 
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTC_LOBE_UTILS_H

#include "Device/includes/LightSampling/LTCs/LTCShading.h"

 /**
 * Some dumb fit (not really precise) to approximate the average Fresnel term over the hemisphere over
 * a microfacet distribution of given roughness, for a given view direction (NoV) and relative_eta.
 */
HIPRT_DEVICE float average_fresnel_fit(float NoV, float roughness, float relative_eta)
{
	constexpr float c0 = 6.24652f;
	constexpr float p0 = 6.76440f;
	constexpr float p1 = -10.07177f;
	constexpr float p2 = -14.93870f;
	constexpr float p3 = -0.26892f;
		
	float F0 = hippt::square((relative_eta - 1.0f) / (relative_eta + 1.0f));

	float k = c0 * roughness * roughness;
	float cos_eff = NoV * (1.0f - k) + k;
	cos_eff = hippt::clamp(0.0f, 1.0f, cos_eff);

	float p = p0 + p1 * roughness + p2 * roughness * roughness + p3 * F0;
	p = hippt::max(p, 1.0e-3f);

	return F0 + (1.0f - F0) * hippt::intrin_pow((1 - cos_eff), p);
}

HIPRT_DEVICE void ltc_lobe_probas(const HIPRTRenderData& render_data,
	float3 vertex_A_worldspace, float3 vertex_B_worldspace, float3 vertex_C_worldspace,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	ColorRGB32F triangle_emission, const DeviceUnpackedEffectiveMaterial& material,
	float& out_coat_proba, float& out_metallic_proba, float& out_specular_proba)
{
	return;
	//float coat_weight = 0.0f;
	//if (material.coat_roughness <= render_data.bsdfs_data.ltcs_data.specular_ltc_maximum_roughness && material.coat > 0.0f)
	//{
	//	float coat_reflected_radiance_triangle_ltc = evaluate_ltc(render_data,
	//		vertex_A_worldspace, vertex_B_worldspace, vertex_C_worldspace,
	//		shading_point, view_direction, shading_normal,
	//		material, LTCLobe::COAT_LOBE) * triangle_emission.luminance();

	//	coat_weight = material.coat * coat_reflected_radiance_triangle_ltc;
	//}

	//float metallic_weight = material.metallic;
	//float specular_weight = 0.0f;

	//if (material.roughness <= render_data.bsdfs_data.ltcs_data.specular_ltc_maximum_roughness && material.specular > 0.0f)
	//{
	//	float specular_radiance_triangle_ltc = evaluate_ltc(render_data,
	//		vertex_A_worldspace, vertex_B_worldspace, vertex_C_worldspace,
	//		shading_point, view_direction, shading_normal,
	//		material, LTCLobe::SPECULAR_LOBE) * triangle_emission.luminance();
	//	
	//	specular_weight = (1.0f - material.metallic) * material.specular * specular_radiance_triangle_ltc;
	//}

	//float diffuse_radiance_triangle_ltc = evaluate_ltc(render_data,
	//	vertex_A_worldspace, vertex_B_worldspace, vertex_C_worldspace,
	//	shading_point, view_direction, shading_normal,
	//	material, LTCLobe::DIFFUSE_LOBE) * triangle_emission.luminance();
	//
	//float diffuse_weight = material.base_color.luminance() * (1.0f - average_fresnel_fit(hippt::dot(view_direction, shading_normal), material.roughness, material.ior)) * diffuse_radiance_triangle_ltc;

	//if (coat_weight + metallic_weight + specular_weight + diffuse_weight == 0.0f)
	//	// All lobes have 0 weight, this is the perfect only-diffuse-lobe case
	//	diffuse_weight = 1.0f;

	///*float metallic = material.metallic;
	//out_metal_1_weight = metallic * outside_object;
	//out_metal_2_weight = metallic * outside_object;

	//float second_roughness_weight = material.second_roughness_weight;
	//out_metal_1_weight = hippt::lerp(out_metal_1_weight, 0.0f, second_roughness_weight);
	//out_metal_2_weight = hippt::lerp(0.0f, out_metal_2_weight, second_roughness_weight);*/

	//float proba_normalize = 1.0f / (coat_weight + metallic_weight + specular_weight + diffuse_weight);

	//out_coat_proba = coat_weight * proba_normalize;
	//out_metallic_proba = metallic_weight * proba_normalize;
	//out_specular_proba = specular_weight * proba_normalize;
}

HIPRT_DEVICE LTCLobeSampleProbabilities ltc_lobe_probas(const HIPRTRenderData& render_data,
	float3 vertex_A_worldspace, float3 vertex_B_worldspace, float3 vertex_C_worldspace,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	ColorRGB32F triangle_emission, const DeviceUnpackedEffectiveMaterial& material)
{
	LTCLobeSampleProbabilities lobe_probabilities;

	ltc_lobe_probas(render_data,
		vertex_A_worldspace, vertex_B_worldspace, vertex_C_worldspace,
		shading_point, view_direction, shading_normal,
		triangle_emission, material,
		lobe_probabilities.coat_proba, lobe_probabilities.metallic_proba, lobe_probabilities.specular_proba);

	return lobe_probabilities;
}

/**
 * Given a potentially multi-layered material, this function samples one of its lobe
 * and returns which lobe was sampled along with its PDF.
 *
 * This is meant for area-light-LTC sampling where sampling all lobes per each shading
 * point would require multiple shadow rays so instead we sample only one lobe
 * stochastically, essentially a one-sample-estimator.
 */
HIPRT_DEVICE LTCLobe ltc_lobe_sample(LTCLobeSampleProbabilities lobe_probabilities, Xorshift32Generator& rng)
{
#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
	return LTCLobe::DIFFUSE_LOBE;
#endif

	float cdf[3];
	cdf[0] = lobe_probabilities.coat_proba;
	cdf[1] = cdf[0] + lobe_probabilities.metallic_proba;
	cdf[2] = cdf[1] + lobe_probabilities.specular_proba;

	float random_number = rng();

	if (random_number < cdf[0])
		return LTCLobe::COAT_LOBE;
	else if (random_number < cdf[1])
		return LTCLobe::METALLIC_LOBE;
	else if (random_number < cdf[2])
		return LTCLobe::SPECULAR_LOBE;
	else
		return LTCLobe::DIFFUSE_LOBE;
}

HIPRT_DEVICE float ltc_lobe_eval_pdf(LTCLobeSampleProbabilities lobe_probabilities, LTCLobe lobe)
{
#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
	if (lobe == LTCLobe::DIFFUSE_LOBE)
		return 1.0f;
	else
		return 0.0f;
#endif
	switch (lobe)
	{
	case COAT_LOBE:
		return lobe_probabilities.coat_proba;

	case METALLIC_LOBE:
		return lobe_probabilities.metallic_proba;

	case SPECULAR_LOBE:
		return lobe_probabilities.specular_proba;

	case DIFFUSE_LOBE:
		return 1.0f - (lobe_probabilities.coat_proba + lobe_probabilities.metallic_proba + lobe_probabilities.specular_proba);

	default:
		return 0.0f;
	}
}

#endif
