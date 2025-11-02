/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SPHERICAL_GAUSSIAN_UTILS_H
#define DEVICE_INCLUDES_LIGHT_TREE_SPHERICAL_GAUSSIAN_UTILS_H

#include "HostDeviceCommon/Math.h"

struct SGLobe
{
	float3 axis;
	float  sharpness;
	float  logAmplitude;
};

 /**
 * Adapted from: https://github.com/yusuketokuyoshi/VSGL
 */

#define SG_LIGHT_SHARPNESS_MAX 2199023255552.0f

 // A dominant visible microfacet normal for the GGX NDF.
 // This normal vector is given by sampling the center of the spherical-cap VNDF [Dupuy and Benyoub 2023 "Sampling Visible GGX Normals with Spherical Caps"].
HIPRT_DEVICE static float3 GGX_dominant_visible_normal(const float3 wi, const float2 roughness)
{
	// Numerically stable implementation for wi.x < 0
	// Similar manner to Tokuyoshi and Eto 2024 "Bounded VNDF Sampling for the Smith-GGX BRDF" Appendix C.
	const float2 v = roughness * make_float2(wi.x, wi.y);
	const float len2 = hippt::dot(v, v);
	const float t = sqrtf(len2 + wi.z * wi.z);
	const float z = wi.z >= 0.0f ? t + wi.z : len2 / (t - wi.z);

	return hippt::normalize(make_float3(roughness.x * roughness.x * wi.x, roughness.y * roughness.y * wi.y, z));
}

// Symmetric GGX using anisotropic alpha roughness.
HIPRT_DEVICE static float SGGX(const float3 m, const float2 roughness)
{
	const float3 stretched = make_float3(m.x / roughness.x, m.y / roughness.y, m.z);
	const float length2 = hippt::dot(stretched, stretched);

	return 1.0f / (M_PI * (roughness.x * roughness.y) * (length2 * length2));
}

// Symmetric GGX using a 2x2 roughness matrix (i.e., Non-axis-aligned GGX w/o the Heaviside function).
HIPRT_DEVICE static float SGGX(const float3 m, const float2x2 roughness_matrix)
{
	const float det = determinant(roughness_matrix);
	const float2x2 roughness_matrix_adjugate = float2x2(roughness_matrix.m[1][1], -roughness_matrix.m[0][1], -roughness_matrix.m[1][0], roughness_matrix.m[0][0]);
	const float length2 = hippt::dot(make_float2(m.x, m.y), roughness_matrix_adjugate * make_float2(m.x, m.y)) / det + m.z * m.z;

	return 1.0f / (M_PI * sqrtf(det) * (length2 * length2));
}

// Reflection lobe based on the symmetric GGX VNDF.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5.2]
HIPRT_DEVICE static float SGGX_reflection_PDF(const float3 wi, const float3 m, const float2x2 roughness_matrix)
{
	return SGGX(m, roughness_matrix) / (4.0f * sqrtf(hippt::dot(make_float2(wi.x, wi.y), roughness_matrix * make_float2(wi.x, wi.y)) + wi.z * wi.z));
}

// Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
// The parameter "cosine" is the cosine of the angle between the SG axis and the pole axis of the hemisphere.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 4]
HIPRT_DEVICE static float VMF_hemispherical_integral(const float cosine, const float sharpness)
{
	// Interpolation factor [Tokuyoshi 2022].
	const float A = 0.6517328826907056171791055021459f;
	const float B = 1.3418280033141287699294252888649f;
	const float C = 7.2216687798956709087860872386955f;
	const float steepness = sharpness * sqrtf((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));
	const float lerp_factor = hippt::clamp(0.0f, 1.0f, 0.5f + 0.5f * (erff(steepness * hippt::clamp(-1.0f, 1.0f, cosine)) / erff(steepness)));

	// Interpolation between upper and lower hemispherical integrals .
	const float e = hippt::intrin_expf(-sharpness);
	return hippt::lerp(e, 1.0f, lerp_factor) / (e + 1.0f);
}

// Exact solution of an SG integral.
HIPRT_DEVICE static float SG_integral(const float sharpness)
{
	return 4.0f * M_PI * hippt::expm1_over_x_fast(-2.0f * sharpness);
}

// Product of two SGs.
HIPRT_DEVICE static SGLobe SG_product(const float3 axis1, const float sharpness1, const float3 axis2, const float sharpness2)
{
	const float3 axis = axis1 * sharpness1 + axis2 * sharpness2;
	const float sharpness = hippt::length(axis);

	const float3 d = axis1 - axis2;
	const float len2 = hippt::dot(d, d);
	const float log_amplitude = -sharpness1 * sharpness2 * len2 / (sharpness + sharpness1 + sharpness2);

	const SGLobe result = { axis / sharpness, sharpness, log_amplitude };

	return result;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 5]
HIPRT_DEVICE static float upper_SG_clamped_cosine_integral_over_two_pi(const float sharpness)
{
	if (sharpness <= 0.5f)
		// Taylor-series approximation for the numerical stability.
		return (((((((-1.0f / 362880.0f) * sharpness + 1.0f / 40320.0f) * sharpness - 1.0f / 5040.0f) * sharpness + 1.0f / 720.0f) * sharpness - 1.0f / 120.0f) * sharpness + 1.0f / 24.0f) * sharpness - 1.0f / 6.0f) * sharpness + 0.5f;

	return (1.0f - hippt::expm1_over_x_fast(-sharpness)) / sharpness;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 6]
HIPRT_DEVICE static float lower_SG_clamped_cosine_integral_over_two_pi(const float sharpness)
{
	const float e = hippt::intrin_expf(-sharpness);

	if (sharpness <= 0.5f)
		// Taylor-series approximation for the numerical stability.
		return e * (((((((((1.0f / 403200.0f) * sharpness - 1.0f / 45360.0f) * sharpness + 1.0f / 5760.0f) * sharpness - 1.0f / 840.0f) * sharpness + 1.0f / 144.0f) * sharpness - 1.0f / 30.0f) * sharpness + 1.0f / 8.0f) * sharpness - 1.0f / 3.0f) * sharpness + 0.5f);

	return e * (hippt::expm1_over_x_fast(-sharpness) - e) / sharpness;
}

// Approximate product integral of an SG and clamped cosine / pi.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 7]
HIPRT_DEVICE static float SG_clamped_cosine_product_integral_over_pi(const float cosine, const float sharpness)
{
	// Fitted approximation for t(sharpness).
	static const float A = 2.7360831611272558028247203765204f;
	static const float B = 17.02129778174187535455530451145f;
	static const float C = 4.0100826728510421403939290030394f;
	static const float D = 15.219156263147210594866010069381f;
	static const float E = 76.087896272360737270901154261082f;
	const float t = sharpness * sqrtf(0.5f * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
	const float tz = t * cosine;

	const float INV_SQRTPI = 0.56418958354775628694807945156077f; // = 1.0f / sqrtf(pi).
	const float lerp_factor = hippt::clamp(0.0f, 1.0f, 0.5f * (cosine * hippt::erfcf_fast(-tz) + hippt::erfcf_fast(t)) - 0.5f * INV_SQRTPI * hippt::intrin_expf(-tz * tz) * hippt::intrin_expm1f(t * t * (cosine * cosine - 1.0f)) / t);

	// Interpolation between lower and upper hemispherical integrals.
	const float lower_integral = lower_SG_clamped_cosine_integral_over_two_pi(sharpness);
	const float upper_integral = upper_SG_clamped_cosine_integral_over_two_pi(sharpness);
	return 2.0f * hippt::lerp(lower_integral, upper_integral, lerp_factor);
}

#endif
