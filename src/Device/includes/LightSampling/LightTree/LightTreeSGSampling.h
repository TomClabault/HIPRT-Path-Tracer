/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/ONB.h"
#include "Device/includes/Sampling.h" // For reflect_ray()
#include "Device/includes/LightSampling/LightTree/SphericalGaussianUtils.h"

#include "HostDeviceCommon/RenderData.h"

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H

HIPRT_DEVICE float light_tree_sg_node_importance(const LightTreeSGNodeDevice& node, float3 shading_point, float3 view_direction, float3 shading_normal, 
	float specular, float alpha_x, float alpha_y)
{
	if (node.total_power == 0.0f)
		return 0.0f;

	/*float3 max_corner;
	max_corner.x = (shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x;
	max_corner.y = (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y;
	max_corner.z = (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z;

	if (hippt::dot(max_corner - shading_point, shading_normal) <= 0.0f)
		return 0.0f;*/

	// Load an SG light.
	const float3 lightVec = node.gaussian_spatial_mean - shading_point;
	const float squaredDistance = hippt::dot(lightVec, lightVec);
	const float3 to_light_direction = lightVec / hippt::sqrt(squaredDistance);

	// Use conservative spatial variance for outliers
	float c = hippt::clamp(0.0f, 1.0f, hippt::dot(shading_normal, -to_light_direction));
	// Clamp the variance for the numerical stability.
	float variance = hippt::max(node.gaussian_spatial_variance, squaredDistance / SG_LIGHT_SHARPNESS_MAX);
	variance = variance * (1.0f - c) + 0.5f * hippt::square(node.bounding_sphere_radius) * c;

	// Compute the maximum emissive radiance of the SG light.
	// (maximum radiant intensity)/(2*pi*variance) where (maximum radiant intensity)/(2*pi) is given by spherical_gaussian_light.intensity.
	// 
	// This value can be precomputed in the SG light generation if we don't clamp the variance.
	//
	// 'emissive' should be divided by SG_integral(node.vmf_sharpness) but this is already baked in
	// node.total_power
	const float emissive = node.total_power / variance;

	// Compute SG sharpness for a light distribution viewed from the shading point.
	const float light_sharpness = squaredDistance / variance;

	// Light lobe given by the product of the light distribution viewed from the shading point and the directional distribution of the SG light.
	const SGLobe lightLobe = SG_product(-node.vmf_axis, node.vmf_sharpness, to_light_direction, light_sharpness);

	// Diffuse SG lighting.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 4]
	const float amplitude = hippt::intrin_expf(lightLobe.logAmplitude);
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

		const float vlen = hippt::sqrt(hippt::square(wi.x) + hippt::square(wi.y));
		const float2 v = (vlen != 0.0f) ? make_float2(wi.x, wi.y) / vlen : make_float2(1.0f, 0.0f);
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
			hippt::is_finite(1.0f + tr + det) ?

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

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = emissive_triangle_index;
	light_sample.pdf = cumulative_probability * (1.0f / current_node.triangle_count); // Sampling that triangle in that node

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
