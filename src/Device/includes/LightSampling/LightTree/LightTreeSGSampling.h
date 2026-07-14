/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/LightSampling/LightTree/SphericalGaussianUtils.h"
#include "Device/includes/ONB.h"
#include "Device/includes/Sampling.h" // For reflect_ray()

#include "HostDeviceCommon/KernelOptions/LightTreeSGOptions.h"
#include "HostDeviceCommon/RenderData.h"

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H

// For precomputing some data that doesn't change for the shading point and thus is the same for the whole traversal of the tree
struct SGSpecularImportanceData
{
	HIPRT_DEVICE SGSpecularImportanceData() = default;
	HIPRT_DEVICE SGSpecularImportanceData(float3_t ws_view_direction, float3_t ws_shading_normal, float alpha_x, float alpha_y)
	{
		build_ONB(ws_shading_normal, T, B);

		wi = world_to_local_frame(T, B, ws_shading_normal, ws_view_direction);

		const float vlen = hippt::sqrt(hippt::square(wi.x) + hippt::square(wi.y));
		const float2_t v = (vlen != 0.0f) ? make_float2(wi.x, wi.y) / vlen : make_float2(1.0f, 0.0f);
		const float2x2 jacobian_matrix =
			float2x2(v.x, -v.y, v.y, v.x) * float2x2(0.5f, 0.0f, 0.0f, 0.5f / wi.z); // Omit abs() unlike the paper since it doesn't affect JJ^T.

		// Compute JJ^T for NDF filtering.
		jj_matrix = jacobian_matrix * transpose(jacobian_matrix);

		// Convert the roughness parameter from slope space to the orthographically projected space.
		// [Tokuyoshi and Kaplanyan 2021 "Stable Geometric Specular Antialiasing with Projected-Space NDF Filtering", Eq. 4]
		const float2_t roughness_2 = make_float2(alpha_x * alpha_x, alpha_y * alpha_y);

		// Preprocess for the lobe visibility.
		// Approximate the reflection lobe with an SG whose axis is a dominant reflection vector.
		// We use a conservative SG sharpness to filter the visibility as mentioned in the last paragraph "Filtered Visibility" of Section 5.2 of the paper.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"]
		// Unlike the paper, we use a dominant visible microfacet normal instead of the shading normal to obtain the dominant reflection vector.
		const float roughness_max_2 = hippt::max(roughness_2.x, roughness_2.y);
		reflection_sharpness		= (1.0f - roughness_max_2) / hippt::max(2.0f * roughness_max_2, hippt::FLOAT_MIN);

		projected_roughness_2 =
			make_float2(roughness_2.x / hippt::max(1.0f - roughness_2.x, 1.0e-8f), roughness_2.y / hippt::max(1.0f - roughness_2.y, 1.0e-8f));
	}

	// Local view direction
	float3_t wi;
	// Local shading frame tangent and bitangent
	float3_t T, B;

	float2x2 jj_matrix;

	float reflection_sharpness;
	float2_t projected_roughness_2;
};

HIPRT_DEVICE float light_tree_sg_node_importance(const LightTreeSGNodeDevice& node,
												 const SGSpecularImportanceData& spec_data,
												 float3_t shading_point,
												 float3_t view_direction,
												 float3_t shading_normal,
												 float specular,
												 float alpha_x,
												 float alpha_y)
{
	if (node.total_power == 0.0f)
		return 0.0f;

	// Commented because too expensive (RIS 16 bistro: 38ms --> 40ms) but not massively better quality (just a little bit)
	/*float3_t max_corner;
	max_corner.x = (shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x;
	max_corner.y = (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y;
	max_corner.z = (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z;

	if (hippt::dot(max_corner - shading_point, shading_normal) <= 0.0f)
		return 0.0f;*/

	// Load an SG light.
	const float3_t lightVec			  = node.gaussian_spatial_mean - shading_point;
	const float squaredDistance		  = hippt::dot(lightVec, lightVec);
	const float3_t to_light_direction = lightVec / hippt::sqrt(squaredDistance);

	// Use conservative spatial variance for outliers
	float c = hippt::clamp(0.0f, 1.0f, hippt::dot(shading_normal, -to_light_direction));
	// Clamp the variance for the numerical stability.
	float variance = hippt::max(node.gaussian_spatial_variance, squaredDistance / SG_LIGHT_SHARPNESS_MAX);
	variance	   = variance * (1.0f - c) + 0.5f * hippt::square(node.bounding_sphere_radius) * c;

	// Compute the maximum emissive radiance of the SG light.
	// (maximum radiant intensity)/(2*pi*variance) where (maximum radiant intensity)/(2*pi) is given by spherical_gaussian_light.intensity.
	//
	// This value can be precomputed in the SG light generation if we don't clamp the variance.
	//
	// 'emissive' should be divided by SG_integral(node.vmf.sharpness) but this is already baked in
	// node.total_power
	const float emissive = node.total_power / variance;

	// Compute SG sharpness for a light distribution viewed from the shading point.
	const float light_sharpness = squaredDistance / variance;

	// Light lobe given by the product of the light distribution viewed from the shading point and the directional distribution of the SG light.
	const SGLobe lightLobe = SG_product(-node.vmf.axis, node.vmf.sharpness, to_light_direction, light_sharpness);

	// Diffuse SG lighting.
	// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 4]
	const float amplitude			 = hippt::intrin_expf(lightLobe.logAmplitude);
	const float cosine				 = hippt::clamp(-1.0f, 1.0f, hippt::dot(lightLobe.axis, shading_normal));
	const float diffuse_illumination = amplitude * SG_clamped_cosine_product_integral_over_pi(cosine, lightLobe.sharpness);

	float specular_illumination = 0.0f;
#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	if (specular)
	{
		// Glossy SG lighting.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5]
		const float light_lobe_variance = 1.0f / lightLobe.sharpness;
		const float2x2 filtered_proj_roughness_mat =
			float2x2(spec_data.projected_roughness_2.x, 0.0f, 0.0f, spec_data.projected_roughness_2.y) + 2.0f * light_lobe_variance * spec_data.jj_matrix;

		// Compute the determinant of JJ^T without catastrophic cancellation.
		const float det_JJ4 = 1.0f / (4.0f * spec_data.wi.z * spec_data.wi.z); // = 4 * determiant(JJ^T).
		// Compute the determinant of filtered_proj_roughness_mat in a numerically stable manner.
		// See the supplementary document (Section 5.2) of the paper for the derivation.
		const float det =
			spec_data.projected_roughness_2.x * spec_data.projected_roughness_2.y +
			2.0f * light_lobe_variance *
				(spec_data.projected_roughness_2.x * spec_data.jj_matrix.m[0][0] + spec_data.projected_roughness_2.y * spec_data.jj_matrix.m[1][1]) +
			light_lobe_variance * light_lobe_variance * det_JJ4;

		// NDF filtering in a numerically stable manner.
		// See the supplementary document (Section 5.2) of the paper for the derivation.
		const float tr = filtered_proj_roughness_mat.m[0][0] + filtered_proj_roughness_mat.m[1][1];
		const float2x2 filtered_roughness_matrix =
			hippt::is_finite(1.0f + tr + det) ? hippt::min(filtered_proj_roughness_mat + float2x2(det, 0.0f, 0.0f, det), hippt::FLOAT_MAX) / (1.0f + tr + det)
											  : float2x2(hippt::min(filtered_proj_roughness_mat.m[0][0], hippt::FLOAT_MAX) /
															 hippt::min(filtered_proj_roughness_mat.m[0][0] + 1.0f, hippt::FLOAT_MAX),
														 0.0f, 0.0f,
														 hippt::min(filtered_proj_roughness_mat.m[1][1], hippt::FLOAT_MAX) /
															 hippt::min(filtered_proj_roughness_mat.m[1][1] + 1.0f, hippt::FLOAT_MAX));

		// Evaluate the filtered reflection lobe.
		const float3_t half_vector_unormalized = spec_data.wi + world_to_local_frame(spec_data.T, spec_data.B, shading_normal, lightLobe.axis);
		const float3_t half_vector			   = half_vector_unormalized / hippt::max(hippt::length(half_vector_unormalized), hippt::FLOAT_MIN);
		const float pdf						   = SGGX_reflection_PDF(spec_data.wi, half_vector, filtered_roughness_matrix);

		const float3_t dominant_normal =
			local_to_world_frame(spec_data.T, spec_data.B, shading_normal, GGX_dominant_visible_normal(spec_data.wi, make_float2(alpha_x, alpha_y)));
		const float3_t reflection_vector = reflect_ray(view_direction, dominant_normal) * spec_data.reflection_sharpness;

		// Visibility of the SG light in the upper hemisphere.
		const float3_t product_vector	 = reflection_vector + lightLobe.axis * lightLobe.sharpness; // Axis of the SG product lobe.
		const float product_sharpness	 = hippt::length(product_vector);
		const float3_t product_direction = product_vector / product_sharpness;
		const float visibility			 = VMF_hemispherical_integral(hippt::dot(product_direction, shading_normal), product_sharpness);

		// Eq. 12 of the paper.
		specular_illumination = amplitude * visibility * pdf * SG_integral(lightLobe.sharpness);
	}
#endif

	// Finally, we multiply the common SG-light coefficient.
	return emissive * (diffuse_illumination + specular * specular_illumination);
}

#if LightTreeSGDoSplitting == KERNEL_OPTION_TRUE

HIPRT_DEVICE float light_tree_sg_node_variance(const LightTreeSGNodeDevice& node, float3_t shading_point)
{
	float3_t node_center		 = node.gaussian_spatial_mean;
	float bounding_sphere_radius = node.bounding_sphere_radius;

	// Compute a and b for the geometric mean and variance
	float a = hippt::max(hippt::length(shading_point - node_center) - bounding_sphere_radius, 1.0e-3f);
	float b = hippt::length(shading_point - node_center) + bounding_sphere_radius;

	float a3 = hippt::pow_3(a);
	float b3 = hippt::pow_3(b);

	float mean_geometric	 = 1.0f / (a * b);
	float variance_geometric = (b3 - a3) / (3.0f * (b - a) * a3 * b3) - 1.0f / (a * a * b * b);
	float variance			 = (node.energy_variance * variance_geometric + node.energy_variance * hippt::square(mean_geometric) +
						hippt::square(node.get_energy_average()) * variance_geometric) *
					 hippt::square(node.total_emitter_count);

	return hippt::sqrt(hippt::sqrt(1.0f / (1.0f + hippt::sqrt(variance))));
}

HIPRT_DEVICE LightSampleArray<LightTreeSGSplittingMaxLightSamples> sample_one_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data,
																											  float3_t shading_point,
																											  float3_t view_direction,
																											  float3_t shading_normal,
																											  float3_t geometric_normal,
																											  const DeviceUnpackedEffectiveMaterial& material,
																											  int last_hit_primitive_index,
																											  Xorshift32Generator& rng)
{
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;

	float material_specular_weight =
		(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float sg_roughness		 = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, material.coat * material.coat_roughness + material.metallic * material.roughness +
																				  material_specular_weight * material.roughness / specular_lobes_sum);
	float sg_anisotropy		 = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy +
						  material_specular_weight * material.anisotropy / specular_lobes_sum;

	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	int stack_pointer												   = 0;
	unsigned int node_index_stack[LightTreeSGSplittingMaxLightSamples] = { 0 };
	unsigned int light_samples_counter								   = 1;

	int node_indices_to_be_traversed_sp									  = 0;
	int node_indices_to_be_traversed[LightTreeSGSplittingMaxLightSamples] = { -1 };

	while (light_samples_counter < LightTreeSGSplittingMaxLightSamples && stack_pointer >= 0)
	{
		unsigned int node_index			   = node_index_stack[stack_pointer--];
		LightTreeSGNodeDevice current_node = nodes[node_index];

		light_samples_counter--;

		float node_importance =
			light_tree_sg_node_importance(current_node, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		if (node_importance > 0.0f)
		{
			float node_variance = light_tree_sg_node_variance(current_node, shading_point);
			if (node_variance < render_data.light_tree_sg.settings.light_tree_sg_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold exceeded, exploring both branches of the tree

				float node_importance_left = light_tree_sg_node_importance(nodes[current_node.left_child_index_or_first_triangle_index], spec_data,
																		   shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
				float node_importance_right =
					light_tree_sg_node_importance(nodes[current_node.left_child_index_or_first_triangle_index + 1], spec_data, shading_point, view_direction,
												  shading_normal, sg_specular_weight, alpha_x, alpha_y);

				if (node_importance_left > node_importance_right)
				{
					// We're going to want to explore the left child first

					// So inserting the right child first
					if (node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
						light_samples_counter++;
					}

					// And then the left child such that the left child is popped first
					// and explored first
					if (stack_pointer < LightTreeSGSplittingMaxLightSamples - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						light_samples_counter++;
					}
				}
				else
				{
					// We're going to want to explore the right child first

					// So inserting the left child first
					if (node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						light_samples_counter++;
					}

					// And then the right child such that the right child is popped first
					// and explored first
					if (stack_pointer < LightTreeSGSplittingMaxLightSamples - 1 && node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
						light_samples_counter++;
					}
				}
			}
			else
			{
				// Variance threshold cool, picking a light from this sub tree
				light_samples_counter++;

				node_indices_to_be_traversed[node_indices_to_be_traversed_sp++] = node_index;
			}
		}
	}

	LightSampleArray<LightTreeSGSplittingMaxLightSamples> light_samples_out;

	while (stack_pointer >= 0)
	{
		unsigned int node_index			   = node_index_stack[stack_pointer--];
		LightTreeSGNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeSGNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

			float left_importance =
				light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
			float right_importance =
				light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);

			float p_left = left_importance / (left_importance + right_importance);
			if (left_importance == 0.0f && right_importance == 0.0f)
			{
				// Indicating no sample
				cumulative_probability = -1.0f;

				break;
			}

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

		if (cumulative_probability != -1.0f)
		{
			int index					= current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index			= render_data.light_tree_sg.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			light_samples_out[light_samples_counter - 1].emissive_triangle_global_index = emissive_triangle_index;
			light_samples_out[light_samples_counter - 1].pdf							= cumulative_probability / current_node.triangle_count;
			light_samples_counter--;
		}

		// We have found a good node in this tree
	}

	// Also stream candidates that we're deemed as not needing splitting
	int to_be_traversed_index = 0;
	while (to_be_traversed_index < node_indices_to_be_traversed_sp)
	{
		unsigned int node_index			   = node_indices_to_be_traversed[to_be_traversed_index++];
		LightTreeSGNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeSGNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

			float left_importance =
				light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
			float right_importance =
				light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);

			float p_left = left_importance / (left_importance + right_importance);
			if (left_importance == 0.0f && right_importance == 0.0f)
			{
				// Indicating no sample
				cumulative_probability = -1.0f;

				break;
			}

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

		if (cumulative_probability != -1.0f)
		{
			// We have found a good node in this tree

			int index					= current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index			= render_data.light_tree_sg.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			light_samples_out[light_samples_counter - 1].emissive_triangle_global_index = emissive_triangle_index;
			light_samples_out[light_samples_counter - 1].pdf							= cumulative_probability / current_node.triangle_count;
			light_samples_counter--;
		}
	}

	return light_samples_out;
}

HIPRT_DEVICE void replay_splitting(const HIPRTRenderData& render_data,
								   const LightTreeSGNodeDevice* nodes,
								   const SGSpecularImportanceData& spec_data,
								   unsigned int node_index,
								   unsigned int& collected_split_samples,
								   float3_t shading_point,
								   float3_t view_direction,
								   float3_t shading_normal,
								   float sg_specular_weight,
								   float alpha_x,
								   float alpha_y)
{
	int stack_pointer												   = 0;
	unsigned int node_index_stack[LightTreeSGSplittingMaxLightSamples] = { node_index };

	while (collected_split_samples < LightTreeSGSplittingMaxLightSamples && stack_pointer >= 0)
	{
		unsigned int node_index			   = node_index_stack[stack_pointer--];
		LightTreeSGNodeDevice current_node = nodes[node_index];

		collected_split_samples--;

		float node_importance =
			light_tree_sg_node_importance(current_node, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		if (node_importance > 0.0f)
		{
			float node_variance = light_tree_sg_node_variance(current_node, shading_point);
			if (node_variance < render_data.light_tree_sg.settings.light_tree_sg_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold exceeded, exploring both branches of the tree

				float node_importance_left = light_tree_sg_node_importance(nodes[current_node.left_child_index_or_first_triangle_index], spec_data,
																		   shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
				float node_importance_right =
					light_tree_sg_node_importance(nodes[current_node.left_child_index_or_first_triangle_index + 1], spec_data, shading_point, view_direction,
												  shading_normal, sg_specular_weight, alpha_x, alpha_y);

				if (node_importance_left > node_importance_right)
				{
					if (node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
						collected_split_samples++;
					}

					if (stack_pointer < LightTreeSGSplittingMaxLightSamples - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						collected_split_samples++;
					}
				}
				else
				{
					if (node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						collected_split_samples++;
					}

					if (stack_pointer < LightTreeSGSplittingMaxLightSamples - 1 && node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
						collected_split_samples++;
					}
				}
			}
			else
				collected_split_samples++;
		}
	}
}

HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data,
														  float3_t shading_point,
														  float3_t view_direction,
														  float3_t shading_normal,
														  const DeviceUnpackedEffectiveMaterial& material,
														  int global_emissive_triangle_index)
{
	if (global_emissive_triangle_index == -1)
		return 0.0f;

	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;
	LightTreeSGNodeDevice current_node = nodes[0];

	float material_specular_weight =
		(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float sg_roughness		 = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, material.coat * material.coat_roughness + material.metallic * material.roughness +
																				  material_specular_weight * material.roughness / specular_lobes_sum);
	float sg_anisotropy		 = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy +
						  material_specular_weight * material.anisotropy / specular_lobes_sum;

	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	float root_node_importance =
		light_tree_sg_node_importance(current_node, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail												 = render_data.light_tree_sg.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth											 = 0;
	unsigned int collected_split_samples								 = 1;
	int split_replay_stack_pointer										 = -1;
	unsigned int split_replay_stack[LightTreeSGSplittingMaxLightSamples] = { 0 };
	// We're going to disable splitting after encountering
	// the first node that isn't split
	bool can_split = true;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance =
			light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		float right_importance =
			light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return 0.0f;

		bool node_split = false;
		if (collected_split_samples < LightTreeSGSplittingMaxLightSamples && can_split)
		{
			float node_variance = light_tree_sg_node_variance(current_node, shading_point);
			node_split			= node_variance < render_data.light_tree_sg.settings.light_tree_sg_splitting_variance && current_node.triangle_count == 0;
		}

		if (node_split)
		{
			collected_split_samples--;

			if (left_importance > right_importance)
			{
				// We're going to want to explore the left child first
				// So inserting the right child first
				if (right_importance > 0.0f)
				{
					split_replay_stack[++split_replay_stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
					collected_split_samples++;
				}

				if (split_replay_stack_pointer < LightTreeSGSplittingMaxLightSamples - 1)
				{
					split_replay_stack[++split_replay_stack_pointer] = current_node.left_child_index_or_first_triangle_index;
					collected_split_samples++;
				}
			}
			else
			{
				if (left_importance > 0.0f)
				{
					split_replay_stack[++split_replay_stack_pointer] = current_node.left_child_index_or_first_triangle_index;
					collected_split_samples++;
				}

				if (split_replay_stack_pointer < LightTreeSGSplittingMaxLightSamples - 1)
				{
					split_replay_stack[++split_replay_stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
					collected_split_samples++;
				}
			}

			bool exploring_left_child_first = left_importance > right_importance;
			bool needs_to_go_right_child	= (bit_trail & (1 << current_depth)) != 0;

			if ((exploring_left_child_first && needs_to_go_right_child) || (!exploring_left_child_first && !needs_to_go_right_child))
				replay_splitting(render_data, nodes, spec_data, split_replay_stack[split_replay_stack_pointer--], collected_split_samples, shading_point,
								 view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);

			// We pushed 2 nodes to the splitting replay stack but
			// just our PDF computation code here is going to go down one
			// of the 2 nodes (the last node we pushed, the one with the largest importance)
			// so we can remove that node from the replay stack
			split_replay_stack_pointer--;
		}
		else
			can_split = false;

		float p_left = left_importance / (left_importance + right_importance);
		if (!(bit_trail & (1 << current_depth)))
		{
			// If the bit is not set we're going to the left
			current_node = left_child;

			if (!node_split)
				cumulative_probability *= p_left;
		}
		else
		{
			current_node = right_child;

			if (!node_split)
				cumulative_probability *= 1.0f - p_left;
		}

		current_depth++;
	}

	// Probability of going down the tree + probability of sampling that triangle in the node
	return cumulative_probability / current_node.triangle_count;
}

#else

HIPRT_DEVICE LightSampleArray<1> sample_one_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data,
																			float3_t shading_point,
																			float3_t view_direction,
																			float3_t shading_normal,
																			float3_t geometric_normal,
																			const DeviceUnpackedEffectiveMaterial& material,
																			int last_hit_primitive_index,
																			Xorshift32Generator& rng)
{
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;

	LightTreeSGNodeDevice current_node = nodes[0];

	float material_specular_weight =
		(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float sg_roughness		 = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, material.coat * material.coat_roughness + material.metallic * material.roughness +
																				  material_specular_weight * material.roughness / specular_lobes_sum);
	float sg_anisotropy		 = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy +
						  material_specular_weight * material.anisotropy / specular_lobes_sum;

	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance =
			light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		float right_importance =
			light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return LightSampleArray<1>{ LightSampleInformation() };

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

	int index					= current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
	int triangle_index			= render_data.light_tree_sg.indices_array[index];
	int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = emissive_triangle_index;
	light_sample.pdf							= cumulative_probability * (1.0f / current_node.triangle_count); // PDF of sampling that triangle in that node

	return LightSampleArray<1>{ light_sample };
}

HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data,
														  float3_t shading_point,
														  float3_t view_direction,
														  float3_t shading_normal,
														  const DeviceUnpackedEffectiveMaterial& material,
														  int global_emissive_triangle_index)
{
	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;

	LightTreeSGNodeDevice current_node = nodes[0];

	float material_specular_weight =
		(1.0f - material.metallic) * (1.0f - material.specular_transmission * (1.0f - material.diffuse_transmission)) * material.specular;

	float specular_lobes_sum = material.coat + material.metallic + material_specular_weight;
	float sg_specular_weight = hippt::max(material.coat, hippt::max(material.metallic, material_specular_weight));
	float sg_roughness		 = hippt::max(MaterialConstants::ROUGHNESS_CLAMP, material.coat * material.coat_roughness + material.metallic * material.roughness +
																				  material_specular_weight * material.roughness / specular_lobes_sum);
	float sg_anisotropy		 = material.coat * material.coat_anisotropy + material.metallic * material.anisotropy +
						  material_specular_weight * material.anisotropy / specular_lobes_sum;

	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);

#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	SGSpecularImportanceData spec_data(view_direction, shading_normal, alpha_x, alpha_y);
#else
	SGSpecularImportanceData spec_data;
#endif

	float root_node_importance =
		light_tree_sg_node_importance(current_node, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail		= render_data.light_tree_sg.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeSGNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance =
			light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
		float right_importance =
			light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x, alpha_y);
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

#endif
