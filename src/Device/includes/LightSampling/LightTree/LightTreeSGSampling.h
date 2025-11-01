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
HIPRT_DEVICE float3 GGX_dominant_visible_normal(const float3 wi, const float2 roughness)
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
HIPRT_DEVICE float SGGX(const float3 m, const float2x2 roughness_matrix)
{
	const float det = hippt::max(determinant(roughness_matrix), hippt::FLOAT_MIN);
	const float2x2 roughness_matrix_adjugate = float2x2(roughness_matrix.m[1][1], -roughness_matrix.m[0][1], -roughness_matrix.m[1][0], roughness_matrix.m[0][0]);
	const float length2 = hippt::dot(make_float2(m.x, m.y), roughness_matrix_adjugate * make_float2(m.x, m.y)) / det + m.z * m.z;

	return 1.0f / (M_PI * sqrtf(det) * (length2 * length2));
}

// Reflection lobe based on the symmetric GGX VNDF.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5.2]
HIPRT_DEVICE float SGGX_reflection_PDF(const float3 wi, const float3 m, const float2x2 roughness_matrix)
{
	return SGGX(m, roughness_matrix) / (4.0f * sqrtf(hippt::dot(make_float2(wi.x, wi.y), roughness_matrix * make_float2(wi.x, wi.y)) + wi.z * wi.z));
}

// Exact solution of an SG integral.
HIPRT_DEVICE float SG_integral(const float sharpness)
{
	return 4.0f * M_PI * hippt::expm1_over_x(-2.0f * sharpness);
}

// Product of two SGs.
HIPRT_DEVICE SGLobe SG_product(const float3 axis1, const float sharpness1, const float3 axis2, const float sharpness2)
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
	const float log_amplitude = -sharpness1 * sharpness2 * len2 / hippt::max(sharpness + sharpness1 + sharpness2, hippt::FLOAT_MIN);

	const SGLobe result = { axis / hippt::max(sharpness, hippt::FLOAT_MIN), sharpness, log_amplitude };

	return result;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 5]
HIPRT_DEVICE float upper_SG_clamped_cosine_integral_over_two_pi(const float sharpness)
{
	if (sharpness <= 0.5f)
		// Taylor-series approximation for the numerical stability.
		return (((((((-1.0f / 362880.0f) * sharpness + 1.0f / 40320.0f) * sharpness - 1.0f / 5040.0f) * sharpness + 1.0f / 720.0f) * sharpness - 1.0f / 120.0f) * sharpness + 1.0f / 24.0f) * sharpness - 1.0f / 6.0f) * sharpness + 0.5f;

	return (1.0f - hippt::expm1_over_x(-sharpness)) / sharpness;
}

// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 6]
HIPRT_DEVICE float lower_SG_clamped_cosine_integral_over_two_pi(const float sharpness)
{
	const float e = expf(-sharpness);

	if (sharpness <= 0.5f)
		// Taylor-series approximation for the numerical stability.
		return e * (((((((((1.0f / 403200.0f) * sharpness - 1.0f / 45360.0f) * sharpness + 1.0f / 5760.0f) * sharpness - 1.0f / 840.0f) * sharpness + 1.0f / 144.0f) * sharpness - 1.0f / 30.0f) * sharpness + 1.0f / 8.0f) * sharpness - 1.0f / 3.0f) * sharpness + 0.5f);

	return e * (hippt::expm1_over_x(-sharpness) - e) / sharpness;
}

// Approximate product integral of an SG and clamped cosine / pi.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 7]
HIPRT_DEVICE float SG_clamped_cosine_product_integral_over_pi(const float cosine, const float sharpness)
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
	const float lerp_factor = hippt::clamp(0.0f, 1.0f, hippt::max(0.5f * (cosine * erfcf(-tz) + erfcf(t)) - 0.5f * INV_SQRTPI * expf(-tz * tz) * expm1f(t * t * (cosine * cosine - 1.0f)) / t, CLAMPING_THRESHOLD));

	// Interpolation between lower and upper hemispherical integrals.
	const float lower_integral = lower_SG_clamped_cosine_integral_over_two_pi(sharpness);
	const float upper_integral = upper_SG_clamped_cosine_integral_over_two_pi(sharpness);
	return 2.0f * hippt::lerp(lower_integral, upper_integral, lerp_factor);
}

// Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
// The parameter "cosine" is the cosine of the angle between the SG axis and the pole axis of the hemisphere.
// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 4]
HIPRT_DEVICE float VMF_hemispherical_integral(const float cosine, const float sharpness)
{
	// Interpolation factor [Tokuyoshi 2022].
	const float A = 0.6517328826907056171791055021459f;
	const float B = 1.3418280033141287699294252888649f;
	const float C = 7.2216687798956709087860872386955f;
	const float steepness = sharpness * sqrtf((0.5f * sharpness + A) / ((sharpness + B) * sharpness + C));
	const float lerp_factor = hippt::clamp(0.0f, 1.0f, 0.5f + 0.5f * (erff(steepness * hippt::clamp(-1.0f, 1.0f, cosine)) / erff(steepness)));

	// Interpolation between upper and lower hemispherical integrals .
	const float e = expf(-sharpness);
	return hippt::lerp(e, 1.0f, lerp_factor) / (e + 1.0f);
}

HIPRT_DEVICE float light_tree_sg_node_importance(const LightTreeSGNodeDevice& node, float3 shading_point, float3 view_direction, float3 shading_normal, 
	float specular, float alpha_x, float alpha_y)
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
	const float3 lightVec = node.gaussian_spatial_mean - shading_point;
	const float squaredDistance = hippt::dot(lightVec, lightVec);
	const float3 to_light_direction = lightVec / sqrtf(squaredDistance);

	// Use conservative spatial variance for outliers
	float c = hippt::clamp(0.0f, 1.0f, hippt::dot(shading_normal, -to_light_direction));
	// Clamp the variance for the numerical stability.
	float variance = hippt::max(node.gaussian_spatial_variance, squaredDistance / SG_LIGHT_SHARPNESS_MAX);
	variance = variance * (1.0f - c) + 0.5f * hippt::square(node.bounding_sphere_radius) * c;

	// Compute the maximum emissive radiance of the SG light.
	// (maximum radiant intensity)/(2*pi*variance) where (maximum radiant intensity)/(2*pi) is given by spherical_gaussian_light.intensity.
	// This value can be precomputed in the SG light generation if we don't clamp the variance.
	const float emissive = node.total_power / (variance * SG_integral(node.vmf_sharpness));

	// Compute SG sharpness for a light distribution viewed from the shading point.
	const float light_sharpness = squaredDistance / variance;

	// Light lobe given by the product of the light distribution viewed from the shading point and the directional distribution of the SG light.
	const SGLobe lightLobe = SG_product(-node.vmf_axis, node.vmf_sharpness, to_light_direction, light_sharpness);

	// Diffuse SG lighting.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 4]
	const float amplitude = expf(lightLobe.logAmplitude);
	const float cosine = hippt::clamp(-1.0f, 1.0f, hippt::dot(lightLobe.axis, shading_normal));
	const float diffuse_illumination = amplitude * SG_clamped_cosine_product_integral_over_pi(cosine, lightLobe.sharpness);

	float specular_illumination = 0.0f;
#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	if (specular)
	{
		float3 T, B;
		build_ONB(shading_normal, T, B);

		// Compute the Jacobian matrix J for the transformation between halfvetors and reflection vectors at halfvector = normal.
		const float3 wi = world_to_local_frame(T, B, shading_normal, view_direction);

		// Convert the roughness parameter from slope space to the orthographically projected space.
		// [Tokuyoshi and Kaplanyan 2021 "Stable Geometric Specular Antialiasing with Projected-Space NDF Filtering", Eq. 4]
		const float2 roughness_2 = make_float2(alpha_x * alpha_x, alpha_y * alpha_y);
		const float2 projected_roughness_2 = make_float2(roughness_2.x / hippt::max(1.0f - roughness_2.x, 1.0e-8f), roughness_2.y / hippt::max(1.0f - roughness_2.y, 1.0e-8f));

		// Preprocess for the lobe visibility.
		// Approximate the reflection lobe with an SG whose axis is a dominant reflection vector.
		// We use a conservative SG sharpness to filter the visibility as mentioned in the last paragraph "Filtered Visibility" of Section 5.2 of the paper.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"]
		// Unlike the paper, we use a dominant visible microfacet normal instead of the shading normal to obtain the dominant reflection vector.
		const float roughness_max_2 = hippt::max(roughness_2.x, roughness_2.y);
		const float reflection_sharpness = (1.0f - roughness_max_2) / hippt::max(2.0f * roughness_max_2, hippt::FLOAT_MIN);

		const float vlen = sqrtf(hippt::square(wi.x) + hippt::square(wi.y));
		const float2 v = (vlen != 0.0f) ? make_float2(wi.x, wi.y) / vlen : float2(1.0f, 0.0f);
		const float2x2 jacobian_matrix = float2x2(v.x, -v.y, v.y, v.x) * float2x2(0.5f, 0.0f, 0.0f, 0.5f / wi.z); // Omit abs() unlike the paper since it doesn't affect JJ^T.

		// Compute JJ^T for NDF filtering.
		const float2x2 jj_matrix = jacobian_matrix * transpose(jacobian_matrix);

		// Compute the determinant of JJ^T without catastrophic cancellation.
		const float det_JJ4 = 1.0f / (4.0f * wi.z * wi.z); // = 4 * determiant(JJ^T).

		// Glossy SG lighting.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5]
		const float light_lobe_variance = 1.0f / lightLobe.sharpness;
		const float2x2 filtered_proj_roughness_mat = float2x2(projected_roughness_2.x, 0.0f, 0.0f, projected_roughness_2.y) + 2.0f * light_lobe_variance * jj_matrix;

		// Compute the determinant of filtered_proj_roughness_mat in a numerically stable manner.
		// See the supplementary document (Section 5.2) of the paper for the derivation.
		const float det = projected_roughness_2.x * projected_roughness_2.y + 2.0f * light_lobe_variance * (projected_roughness_2.x * jj_matrix.m[0][0] + projected_roughness_2.y * jj_matrix.m[1][1]) + light_lobe_variance * light_lobe_variance * det_JJ4;

		// NDF filtering in a numerically stable manner.
		// See the supplementary document (Section 5.2) of the paper for the derivation.
		const float tr = filtered_proj_roughness_mat.m[0][0] + filtered_proj_roughness_mat.m[1][1];
		const float2x2 filtered_roughness_matrix =
			isfinite(1.0f + tr + det) ?

			hippt::min(filtered_proj_roughness_mat + float2x2(det, 0.0f, 0.0f, det), hippt::FLOAT_MAX) / (1.0f + tr + det) :

			float2x2(
				hippt::min(filtered_proj_roughness_mat.m[0][0], hippt::FLOAT_MAX) / hippt::min(filtered_proj_roughness_mat.m[0][0] + 1.0f, hippt::FLOAT_MAX),
				0.0f,
				0.0f,
				hippt::min(filtered_proj_roughness_mat.m[1][1], hippt::FLOAT_MAX) / hippt::min(filtered_proj_roughness_mat.m[1][1] + 1.0f, hippt::FLOAT_MAX));


		// Evaluate the filtered reflection lobe.
		const float3 half_vector_unormalized = wi + world_to_local_frame(T, B, shading_normal, lightLobe.axis);
		const float3 half_vector = half_vector_unormalized / hippt::max(hippt::length(half_vector_unormalized), hippt::FLOAT_MIN);
		const float pdf = SGGX_reflection_PDF(wi, half_vector, filtered_roughness_matrix);

		const float3 dominant_normal = local_to_world_frame(T, B, shading_normal, GGX_dominant_visible_normal(wi, make_float2(alpha_x, alpha_y)));
		const float3 reflection_vector = reflect_ray(view_direction, dominant_normal) * reflection_sharpness;

		// Visibility of the SG light in the upper hemisphere.
		const float3 product_vector = reflection_vector + lightLobe.axis * lightLobe.sharpness; // Axis of the SG product lobe.
		const float product_sharpness = hippt::length(product_vector);
		const float3 product_direction = product_vector / product_sharpness;
		const float visibility = VMF_hemispherical_integral(hippt::dot(product_direction, shading_normal), product_sharpness);

		// Eq. 12 of the paper.
		specular_illumination = amplitude * visibility * pdf * SG_integral(lightLobe.sharpness);
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

	float material_specular_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float sg_roughness = material.coat * material.coat_roughness + material.metallic * material.roughness + material_specular_weight * material.roughness / specular_lobes_sum;
	float sg_anisotropy = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy + material_specular_weight * material.anisotropy / specular_lobes_sum;

	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance = light_tree_sg_node_importance(left_child, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		float right_importance = light_tree_sg_node_importance(right_child, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
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

	float material_specular_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float sg_roughness = material.coat * material.coat_roughness + material.metallic * material.roughness + material_specular_weight * material.roughness / specular_lobes_sum;
	float sg_anisotropy = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy + material_specular_weight * material.anisotropy / specular_lobes_sum;

	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);

	float root_node_importance = light_tree_sg_node_importance(current_node, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail = render_data.light_tree_sg.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance = light_tree_sg_node_importance(left_child, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		float right_importance = light_tree_sg_node_importance(right_child, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
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
