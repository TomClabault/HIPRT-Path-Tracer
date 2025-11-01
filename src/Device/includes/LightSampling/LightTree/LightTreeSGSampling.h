/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/ONB.h"
#include "Device/includes/Sampling.h" // For reflect_ray()

#include "HostDeviceCommon/RenderData.h"

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H

 /**
  * Adapted from: https://github.com/yusuketokuyoshi/VSGL
  */

#define SG_LIGHT_SHARPNESS_MAX 2199023255552.0f

// A dominant visible microfacet normal for the GGX NDF.
// This normal vector is given by sampling the center of the spherical-cap VNDF [Dupuy and Benyoub 2023 "Sampling Visible GGX Normals with Spherical Caps"].
HIPRT_DEVICE float3 GGXDominantVisibleNormal(const float3 wi, const float2 roughness)
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
HIPRT_DEVICE float SGGX(const float3 m, const float2 roughness)
{
	const float3 stretched = make_float3(m.x / roughness.x, m.y / roughness.y, m.z);
	const float length2 = hippt::dot(stretched, stretched);

	return 1.0f / (M_PI * (roughness.x * roughness.y) * (length2 * length2));
}

// Symmetric GGX using a 2x2 roughness matrix (i.e., Non-axis-aligned GGX w/o the Heaviside function).
HIPRT_DEVICE float SGGX(const float3 m, const float2x2 roughnessMat)
{
	const float det = hippt::max(determinant(roughnessMat), hippt::FLOAT_MIN); // TODO: Use Kahan's algorithm for precise determinant [https://pharr.org/matt/blog/2019/11/03/difference-of-floats].
	const float2x2 roughnessMatAdj = float2x2(roughnessMat.m[1][1], -roughnessMat.m[0][1], -roughnessMat.m[1][0], roughnessMat.m[0][0]);
	const float length2 = hippt::dot(make_float2(m.x, m.y), roughnessMatAdj * make_float2(m.x, m.y)) / det + m.z * m.z; // TODO: Use Kahan's algorithm for precise mul and dot [https://pharr.org/matt/blog/2019/11/03/difference-of-floatshttps://pharr.org/matt/blog/2019/11/03/difference-of-floats].

	return 1.0f / (M_PI * sqrtf(det) * (length2 * length2));
}

// Reflection lobe based on the symmetric GGX VNDF.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5.2]
HIPRT_DEVICE float SGGXReflectionPDF(const float3 wi, const float3 m, const float2x2 roughnessMat)
{
	return SGGX(m, roughnessMat) / (4.0f * sqrtf(hippt::dot(make_float2(wi.x, wi.y), roughnessMat * make_float2(wi.x, wi.y)) + wi.z * wi.z));
}

// Exact solution of an SG integral.
HIPRT_DEVICE float SGIntegral(const float sharpness)
{
	return 4.0f * M_PI * hippt::expm1_over_x(-2.0f * sharpness);
}

// Product of two SGs.
HIPRT_DEVICE SGLobe SGProduct(const float3 axis1, const float sharpness1, const float3 axis2, const float sharpness2)
{
	const float3 axis = axis1 * sharpness1 + axis2 * sharpness2;
	const float sharpness = hippt::length(axis);

	// Compute logAmplitude = sharpness - (sharpness1 + sharpness2).
	// Since sharpness - sharpness1 - sharpness2 in floating point arithmetic can produce a significant numerical error, we use a numerically stable form derived by
	// logAmplitude = sharpness - (sharpness1 + sharpness2)
	//              = (||axis1 * sharpness1 + axis2 * sharpness2||^2 - (sharpness1 + sharpness2)^2) / (sharpness + sharpness1 + sharpness2)
	//              = (sharpness1^2 + 2 * sharpness1 * sharpness2 * dot(axis1, axis2) + sharpness2^2 - (sharpness1^2 + 2 * sharpness1 * sharpness2 + sharpness2^2) / (sharpness + sharpness1 + sharpness2)
	//              = 2 * sharpness1 * sharpness2 * (dot(axis1, axis2) - 1) / (sharpness + sharpness1 + sharpness2)
	//              = -sharpness1 * sharpness2 * ||axis1 - axis2||^2 / (sharpness + sharpness1 + sharpness2).
	const float3 d = axis1 - axis2;
	const float len2 = hippt::dot(d, d); // -0.5 * len2 = dot(axis1, axis2) - 1. Using len2 improves the numerical stability when axis1 \approx axis2.
	const float logAmplitude = -sharpness1 * sharpness2 * len2 / hippt::max(sharpness + sharpness1 + sharpness2, hippt::FLOAT_MIN);

	const SGLobe result = { axis / hippt::max(sharpness, hippt::FLOAT_MIN), sharpness, logAmplitude };

	return result;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 5]
HIPRT_DEVICE float UpperSGClampedCosineIntegralOverTwoPi(const float sharpness)
{
	if (sharpness <= 0.5f)
		// Taylor-series approximation for the numerical stability.
		return (((((((-1.0f / 362880.0f) * sharpness + 1.0f / 40320.0f) * sharpness - 1.0f / 5040.0f) * sharpness + 1.0f / 720.0f) * sharpness - 1.0f / 120.0f) * sharpness + 1.0f / 24.0f) * sharpness - 1.0f / 6.0f) * sharpness + 0.5f;

	return (1.0f - hippt::expm1_over_x(-sharpness)) / sharpness;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 6]
HIPRT_DEVICE float LowerSGClampedCosineIntegralOverTwoPi(const float sharpness)
{
	const float e = expf(-sharpness);

	if (sharpness <= 0.5f)
		// Taylor-series approximation for the numerical stability.
		return e * (((((((((1.0f / 403200.0f) * sharpness - 1.0f / 45360.0f) * sharpness + 1.0f / 5760.0f) * sharpness - 1.0f / 840.0f) * sharpness + 1.0f / 144.0f) * sharpness - 1.0f / 30.0f) * sharpness + 1.0f / 8.0f) * sharpness - 1.0f / 3.0f) * sharpness + 0.5f);

	return e * (hippt::expm1_over_x(-sharpness) - e) / sharpness;
}

// Approximate product integral of an SG and clamped cosine / pi.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 7]
HIPRT_DEVICE float SGClampedCosineProductIntegralOverPi2024(const float cosine, const float sharpness)
{
	// Fitted approximation for t(sharpness).
	const float A = 2.7360831611272558028247203765204f;
	const float B = 17.02129778174187535455530451145f;
	const float C = 4.0100826728510421403939290030394f;
	const float D = 15.219156263147210594866010069381f;
	const float E = 76.087896272360737270901154261082f;
	const float t = sharpness * sqrtf(0.5f * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E));
	const float tz = t * cosine;

	// In this HLSL implementation, we roughly implement erfc(x) = 1 - erf(x) which can have a numerical error for large x.
	// Therefore, unlike the original impelemntation [Tokuyoshi et al. 2024], we clamp the lerp factor with the machine epsilon / 2 for a conservative approximation.
	// This clamping is unnecessary for languages that have a precise erfc function (e.g., C++).
	// The original implementation [Tokuyoshi et al. 2024] uses a precise erfc function and does not clamp the lerp factor.
	const float INV_SQRTPI = 0.56418958354775628694807945156077f; // = 1/sqrt(pi).
	const float CLAMPING_THRESHOLD = 0.5f * hippt::FLOAT_EPSILON; // Set zero if a precise erfc function is available.
	const float lerpFactor = hippt::clamp(0.0f, 1.0f, hippt::max(0.5f * (cosine * erfcf(-tz) + erfcf(t)) - 0.5f * INV_SQRTPI * expf(-tz * tz) * expm1f(t * t * (cosine * cosine - 1.0f)) / t, CLAMPING_THRESHOLD));

	// Interpolation between lower and upper hemispherical integrals.
	const float lowerIntegral = LowerSGClampedCosineIntegralOverTwoPi(sharpness);
	const float upperIntegral = UpperSGClampedCosineIntegralOverTwoPi(sharpness);
	return 2.0f * hippt::lerp(lowerIntegral, upperIntegral, lerpFactor);
}

// Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
// The parameter "cosine" is the cosine of the angle between the SG axis and the pole axis of the hemisphere.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 4]
HIPRT_DEVICE float VMFHemisphericalIntegral(const float cosine, const float sharpness)
{
	// Interpolation factor [Tokuyoshi 2022].
	const float A = 0.6517328826907056171791055021459f;
	const float B = 1.3418280033141287699294252888649f;
	const float C = 7.2216687798956709087860872386955f;
	const float steepness = sharpness * sqrtf((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));
	const float lerpFactor = hippt::clamp(0.0f, 1.0f, 0.5f + 0.5f * (erff(steepness * hippt::clamp(-1.0f, 1.0f, cosine)) / erff(steepness)));

	// Interpolation between upper and lower hemispherical integrals .
	const float e = expf(-sharpness);
	return hippt::lerp(e, 1.0f, lerpFactor) / (e + 1.0f);
}

struct ASGLobe
{
	float3 x;
	float3 y;
	float3 z;
	float2 sharpness;
	float  logAmplitude;
};

HIPRT_DEVICE float ASGSharpnessToSGSharpness(const float2 sharpness)
{
	return 2.0 * sqrt(sharpness.x * sharpness.y);
}

// Approximate the reflection lobe with an ASG lobe for microfacet BRDFs.
// This implementation is specialized for isotropic NDFs.
// For a general form for anisotropic NDFs, please see [Xu et al. 2012 "Anisotropic Spherical Gaussians"].
HIPRT_DEVICE ASGLobe ASGReflectionLobe(const float3 dir, const float3 normal, const float roughness2)
{
	// Compute ASG sharpness for the NDF.
	// Unlike Xu et al. [2012], we use the following equation based on the Appendix of [Tokuyoshi and Harada 2019 "Hierarchical Russian Roulette for Vertex Connections"].
	const float sharpnessNDF = 1.0f / roughness2 - 1.0f;

	// Compute a 2x2 Jacobian matrix for the transformation from halfvectors to reflection vectors.
	// Since this matrix is diagonal at the perfect reflection vector, we use only diagonal entries.
	const float2 jacobianDiag = { 2.0f * hippt::dot(dir, normal), 2.0f };

	// Compute the sharpness and axes for the reflection lobe.
	const float2 sharpness = sharpnessNDF / (jacobianDiag * jacobianDiag);
	const float3 axisX = hippt::normalize(hippt::cross(dir, normal));
	const float3 axisZ = reflect_ray(dir, normal);
	const float3 axisY = hippt::cross(axisZ, axisX);

	const ASGLobe result = { axisX, axisY, axisZ, sharpness, 0.0f };
	return result;
}

HIPRT_DEVICE float light_tree_sg_node_importance(const LightTreeSGNodeDevice& node, float3 shading_point, float3 view_direction, float3 shading_normal, float2 roughness, float specular)
{
	if (node.total_power == 0.0f)
		return 0.0f;

	float3 max_corner;
	max_corner.x = (shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x;
	max_corner.y = (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y;
	max_corner.z = (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z;

	if (hippt::dot(max_corner - shading_point, shading_normal) <= 0.0f)
		return 0.0f;

	// Load an SG light.
	const SGLight sgLight = node.to_spherical_gaussian_light();
	const float3 lightVec = sgLight.position - shading_point;
	const float squaredDistance = hippt::dot(lightVec, lightVec);
	const float3 to_light_direction = lightVec / sqrtf(squaredDistance);

	// Use conservative spatial variance for outliers
	float c = hippt::clamp(0.0f, 1.0f, hippt::dot(shading_normal, -to_light_direction));
	// Clamp the variance for the numerical stability.
	float variance = hippt::max(sgLight.variance, squaredDistance / SG_LIGHT_SHARPNESS_MAX);
	variance = variance * (1.0f - c) + 0.5f * hippt::square(node.bounding_sphere_radius) * c;

	// Compute the maximum emissive radiance of the SG light.
	// (maximum radiant intensity)/(2*pi*variance) where (maximum radiant intensity)/(2*pi) is given by sgLight.intensity.
	// This value can be precomputed in the SG light generation if we don't clamp the variance.
	const float emissive = sgLight.power / (variance * SGIntegral(sgLight.sharpness));

	// Compute SG sharpness for a light distribution viewed from the shading point.
	const float lightSharpness = squaredDistance / variance;

	// Light lobe given by the product of the light distribution viewed from the shading point and the directional distribution of the SG light.
	const SGLobe lightLobe = SGProduct(sgLight.axis, sgLight.sharpness, to_light_direction, lightSharpness);

	// Diffuse SG lighting.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 4]
	const float amplitude = expf(lightLobe.logAmplitude);
	const float cosine = hippt::clamp(-1.0f, 1.0f, hippt::dot(lightLobe.axis, shading_normal));
	const float diffuse_illumination = amplitude * SGClampedCosineProductIntegralOverPi2024(cosine, lightLobe.sharpness);

	float specular_illumination = 0.0f;

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	{
		float3 T, B;
		build_ONB(shading_normal, T, B);

		// Compute the Jacobian matrix J for the transformation between halfvetors and reflection vectors at halfvector = normal.
		const float3 wi = world_to_local_frame(T, B, shading_normal, view_direction);

		// Convert the roughness parameter from slope space to the orthographically projected space.
		// [Tokuyoshi and Kaplanyan 2021 "Stable Geometric Specular Antialiasing with Projected-Space NDF Filtering", Eq. 4]
		const float2 roughness2 = roughness * roughness;
		const float2 projRoughness2 = make_float2(roughness2.x / hippt::max(1.0f - roughness2.x, 1.0e-8f), roughness2.y / hippt::max(1.0f - roughness2.y, 1.0e-8f));

		// Preprocess for the lobe visibility.
		// Approximate the reflection lobe with an SG whose axis is a dominant reflection vector.
		// We use a conservative SG sharpness to filter the visibility as mentioned in the last paragraph "Filtered Visibility" of Section 5.2 of the paper.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"]
		// Unlike the paper, we use a dominant visible microfacet normal instead of the shading normal to obtain the dominant reflection vector.
		const float roughnessMax2 = hippt::max(roughness2.x, roughness2.y);
		const float reflecSharpness = (1.0f - roughnessMax2) / hippt::max(2.0f * roughnessMax2, hippt::FLOAT_MIN);

		const float vlen = sqrtf(hippt::square(wi.x) + hippt::square(wi.y));
		const float2 v = (vlen != 0.0f) ? make_float2(wi.x, wi.y) / vlen : float2(1.0f, 0.0f);
		const float2x2 jacobianMat = float2x2(v.x, -v.y, v.y, v.x) * float2x2(0.5f, 0.0f, 0.0f, 0.5f / wi.z); // Omit abs() unlike the paper since it doesn't affect JJ^T.

		// Compute JJ^T for NDF filtering.
		const float2x2 jjMat = jacobianMat * transpose(jacobianMat);

		// Compute the determinant of JJ^T without catastrophic cancellation.
		const float detJJ4 = 1.0f / (4.0f * wi.z * wi.z); // = 4 * determiant(JJ^T).

		// Glossy SG lighting.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5]
		const float lightLobeVariance = 1.0f / lightLobe.sharpness;
		const float2x2 filteredProjRoughnessMat = float2x2(projRoughness2.x, 0.0f, 0.0f, projRoughness2.y) + 2.0f * lightLobeVariance * jjMat;

		// Compute the determinant of filteredProjRoughnessMat in a numerically stable manner.
		// See the supplementary document (Section 5.2) of the paper for the derivation.
		const float det = projRoughness2.x * projRoughness2.y + 2.0f * lightLobeVariance * (projRoughness2.x * jjMat.m[0][0] + projRoughness2.y * jjMat.m[1][1]) + lightLobeVariance * lightLobeVariance * detJJ4;

		// NDF filtering in a numerically stable manner.
		// See the supplementary document (Section 5.2) of the paper for the derivation.
		const float tr = filteredProjRoughnessMat.m[0][0] + filteredProjRoughnessMat.m[1][1];
		const float2x2 filteredRoughnessMat =
			isfinite(1.0f + tr + det) ?

			hippt::min(filteredProjRoughnessMat + float2x2(det, 0.0f, 0.0f, det), hippt::FLOAT_MAX) / (1.0f + tr + det) :

			float2x2(
				hippt::min(filteredProjRoughnessMat.m[0][0], hippt::FLOAT_MAX) / hippt::min(filteredProjRoughnessMat.m[0][0] + 1.0f, hippt::FLOAT_MAX),
				0.0f,
				0.0f,
				hippt::min(filteredProjRoughnessMat.m[1][1], hippt::FLOAT_MAX) / hippt::min(filteredProjRoughnessMat.m[1][1] + 1.0f, hippt::FLOAT_MAX));


		// Evaluate the filtered reflection lobe.
		const float3 halfvecUnormalized = wi + world_to_local_frame(T, B, shading_normal, lightLobe.axis);
		const float3 halfvec = halfvecUnormalized / hippt::max(hippt::length(halfvecUnormalized), hippt::FLOAT_MIN);
		const float pdf = SGGXReflectionPDF(wi, halfvec, filteredRoughnessMat);

#if 1
		const ASGLobe specularLobe = ASGReflectionLobe(view_direction, shading_normal, roughness.x * roughness.y); // Assume roughness.x == roughness.y
		const float3 reflecVec = specularLobe.z * ASGSharpnessToSGSharpness(specularLobe.sharpness);
#else
#if 1
		const float3 dominantNormal = local_to_world_frame(T, B, shading_normal, GGXDominantVisibleNormal(wi, roughness));
#else
		const float3 dominantNormal = normal; // Used in the paper.
#endif
		const float3 reflecVec = reflect_ray(-view_direction, dominantNormal) * reflecSharpness;
#endif

		// Visibility of the SG light in the upper hemisphere.
		const float3 prodVec = reflecVec + lightLobe.axis * lightLobe.sharpness; // Axis of the SG product lobe.
		const float prodSharpness = hippt::length(prodVec);
		const float3 prodDir = prodVec / prodSharpness;
		const float visibility = VMFHemisphericalIntegral(hippt::dot(prodDir, shading_normal), prodSharpness);

		// Eq. 12 of the paper.
		specular_illumination = amplitude * visibility * pdf * SGIntegral(lightLobe.sharpness);
	}
#endif

	// Finally, we multiply the common SG-light coefficient.
	return emissive * (diffuse_illumination + specular * specular_illumination);
}

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal,
	const DeviceUnpackedEffectiveMaterial& material,
	int last_hit_primitive_index,
	Xorshift32Generator& rng)
{
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;

	LightTreeSGNodeDevice current_node = nodes[0];

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance = light_tree_sg_node_importance(left_child, shading_point, view_direction, shading_normal, make_float2(material.roughness, material.roughness), material.specular);
		float right_importance = light_tree_sg_node_importance(right_child, shading_point, view_direction, shading_normal, make_float2(material.roughness, material.roughness), material.specular);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return LightSampleInformation();

		float p_left = left_importance / (left_importance + right_importance);

		if (rng() < p_left)
		{
			current_node = left_child;

			cumulative_probability *= p_left;
		}
		else
		{
			current_node = right_child;

			cumulative_probability *= 1.0f - p_left;
		}
	}

	int index = current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
	int triangle_index = render_data.light_tree_sg.indices_array[index];
	int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
	light_sample.area_measure_pdf *= cumulative_probability;
	light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

	return light_sample;
}

HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data, float3 shading_point, float3 view_direction, float3 shading_normal, 
	const DeviceUnpackedEffectiveMaterial& material,
	int global_emissive_triangle_index)
{
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;

	LightTreeSGNodeDevice current_node = nodes[0];

	float root_node_importance = light_tree_sg_node_importance(current_node, shading_point, view_direction, shading_normal, make_float2(material.roughness, material.roughness), material.specular);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail = render_data.light_tree_sg.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance = light_tree_sg_node_importance(left_child, shading_point, view_direction, shading_normal, make_float2(material.roughness, material.roughness), material.specular);
		float right_importance = light_tree_sg_node_importance(right_child, shading_point, view_direction, shading_normal, make_float2(material.roughness, material.roughness), material.specular);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return 0.0f;

		float p_left = left_importance / (left_importance + right_importance);
		if (!(bit_trail & (1 << current_depth)))
		{
			// If the bit is not set we're going to the left
			current_node = left_child;

			cumulative_probability *= p_left;
		}
		else
		{
			current_node = right_child;

			cumulative_probability *= 1.0f - p_left;
		}

		current_depth++;
	}

	// Probability of going down the tree + probability of sampling that triangle in the node
	return cumulative_probability * 1.0f / (current_node.triangle_count);
}

#endif
