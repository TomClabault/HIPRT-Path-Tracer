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

		float vlen = hippt::sqrt(hippt::square(wi.x) + hippt::square(wi.y));
		float2_t v = (vlen != 0.0f) ? make_float2(wi.x, wi.y) / vlen : make_float2(1.0f, 0.0f);
		float2x2 jacobian_matrix =
			float2x2(v.x, -v.y, v.y, v.x) * float2x2(0.5f, 0.0f, 0.0f, 0.5f / wi.z); // Omit abs() unlike the paper since it doesn't affect JJ^T.

		// Compute JJ^T for NDF filtering.
		jj_matrix = jacobian_matrix * transpose(jacobian_matrix);

		// Convert the roughness parameter from slope space to the orthographically projected space.
		// [Tokuyoshi and Kaplanyan 2021 "Stable Geometric Specular Antialiasing with Projected-Space NDF Filtering", Eq. 4]
		float2_t roughness_2 = make_float2(alpha_x * alpha_x, alpha_y * alpha_y);

		// Preprocess for the lobe visibility.
		// Approximate the reflection lobe with an SG whose axis is a dominant reflection vector.
		// We use a conservative SG sharpness to filter the visibility as mentioned in the last paragraph "Filtered Visibility" of Section 5.2 of the paper.
		// [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting"]
		// Unlike the paper, we use a dominant visible microfacet normal instead of the shading normal to obtain the dominant reflection vector.
		float roughness_max_2 = hippt::max(roughness_2.x, roughness_2.y);
		reflection_sharpness  = (1.0f - roughness_max_2) / hippt::max(2.0f * roughness_max_2, hippt::FLOAT_MIN);

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

struct SGImportanceDebug
{
	float squared_distance;
	float emitter_facing;
	float effective_spatial_variance;
	float sharpness_clamp_variance;
	float final_variance;
	float emissive;
	float light_sharpness;
	float product_log_amplitude;
	float product_sharpness;
	float product_cosine;
	float amplitude;
	float diffuse_integral;
	float final_importance;
	float max_corner_dot;
};

HIPRT_DEVICE float light_tree_sg_node_max_emitter_cosine(const LightTreeSGNodeDevice& node, float3_t shading_point)
{
	float3_t center_to_shading = shading_point - node.gaussian_spatial_mean;
	float distance			   = hippt::length(center_to_shading);
	float support_radius	   = node.bounding_sphere_radius;

	/*
	 * Positional direction cone.
	 *
	 * If the shading point lies inside the bounding sphere, emitters in the node may be seen in arbitrary directions. We cannot reject it.
	 */
	float radius = support_radius;
	if (distance <= radius)
		return 1.0f;

	float sin_theta_u = hippt::clamp(0.0f, 1.0f, radius / distance);
	float cos_theta_u = hippt::sqrt(hippt::max(0.0f, 1.0f - sin_theta_u * sin_theta_u));

	/*
	 * Expanded cone:
	 *
	 * theta_bound = theta_o + theta_u
	 */
	float cos_bound = node.cos_theta_o * cos_theta_u - node.sin_theta_o * sin_theta_u;
	float sin_bound = node.sin_theta_o * cos_theta_u + node.cos_theta_o * sin_theta_u;

	/*
	 * If theta_o + theta_u >= pi, every orientation relative to the
	 * shading point is possible.
	 *
	 * For angles in [0, 2*pi], negative sin indicates that the sum
	 * passed pi. This assumes theta_o is stored in [0, pi].
	 */
	if (sin_bound <= 0.0f)
		return 1.0f;

	float3_t direction = center_to_shading / distance;
	float cos_theta	   = hippt::clamp(-1.0f, 1.0f, hippt::dot(node.orientation_axis, direction));

	// Query direction lies inside the expanded orientation cone.
	if (cos_theta >= cos_bound)
		return 1.0f;

	float sin_theta = hippt::sqrt(hippt::max(0.0f, 1.0f - cos_theta * cos_theta));

	// cos(theta - (theta_o + theta_u))
	return cos_theta * cos_bound + sin_theta * sin_bound;
}

HIPRT_DEVICE float light_tree_sg_evaluate_spatial_lobe(const SpatialSGLobeDevice& spatial_lobe,
													   const VMF& directional_vmf,
													   const SGSpecularImportanceData& spec_data,
													   float3_t shading_point,
													   float3_t view_direction,
													   float3_t shading_normal,
													   float specular,
													   float alpha_x,
													   float alpha_y)
{
	if (spatial_lobe.power <= 0.0f)
		return 0.0f;

	float3_t shading_to_lobe	= spatial_lobe.mean - shading_point;
	float squared_distance		= hippt::dot(shading_to_lobe, shading_to_lobe);
	float safe_squared_distance = hippt::max(squared_distance, 1.0e-20f);
	float3_t to_light_direction = shading_to_lobe * hippt::rsqrt(safe_squared_distance);

	float c						   = hippt::clamp(0.0f, 1.0f, hippt::dot(shading_normal, -to_light_direction));
	float sharpness_clamp_variance = safe_squared_distance / SG_LIGHT_SHARPNESS_MAX;
	float variance				   = hippt::max(spatial_lobe.variance, sharpness_clamp_variance);
	variance					   = variance * (1.0f - c) + 0.5f * hippt::square(spatial_lobe.support_radius) * c;
	variance					   = hippt::max(variance, 1.0e-20f);

	float emissive		  = spatial_lobe.power / variance;
	float light_sharpness = safe_squared_distance / variance;
	SGLobe light_lobe	  = SG_product(-directional_vmf.axis, directional_vmf.sharpness, to_light_direction, light_sharpness);

	float amplitude			   = hippt::intrin_expf(light_lobe.logAmplitude);
	float cosine			   = hippt::clamp(-1.0f, 1.0f, hippt::dot(light_lobe.axis, shading_normal));
	float diffuse_illumination = amplitude * SG_clamped_cosine_product_integral_over_pi(cosine, light_lobe.sharpness);

	float specular_illumination = 0.0f;
#if LightTreeSGDoSpecularImportance == KERNEL_OPTION_TRUE && BSDFOverride != BSDF_LAMBERTIAN && BSDFOverride != BSDF_OREN_NAYAR
	if (specular)
	{
		float light_lobe_variance = 1.0f / light_lobe.sharpness;
		float2x2 filtered_proj_roughness_mat =
			float2x2(spec_data.projected_roughness_2.x, 0.0f, 0.0f, spec_data.projected_roughness_2.y) + 2.0f * light_lobe_variance * spec_data.jj_matrix;

		float det_JJ4 = 1.0f / (4.0f * spec_data.wi.z * spec_data.wi.z);
		float det	  = spec_data.projected_roughness_2.x * spec_data.projected_roughness_2.y +
					2.0f * light_lobe_variance *
						(spec_data.projected_roughness_2.x * spec_data.jj_matrix.m[0][0] + spec_data.projected_roughness_2.y * spec_data.jj_matrix.m[1][1]) +
					light_lobe_variance * light_lobe_variance * det_JJ4;

		float tr = filtered_proj_roughness_mat.m[0][0] + filtered_proj_roughness_mat.m[1][1];
		float2x2 filtered_roughness_matrix =
			hippt::is_finite(1.0f + tr + det) ? hippt::min(filtered_proj_roughness_mat + float2x2(det, 0.0f, 0.0f, det), hippt::FLOAT_MAX) / (1.0f + tr + det)
											  : float2x2(hippt::min(filtered_proj_roughness_mat.m[0][0], hippt::FLOAT_MAX) /
															 hippt::min(filtered_proj_roughness_mat.m[0][0] + 1.0f, hippt::FLOAT_MAX),
														 0.0f, 0.0f,
														 hippt::min(filtered_proj_roughness_mat.m[1][1], hippt::FLOAT_MAX) /
															 hippt::min(filtered_proj_roughness_mat.m[1][1] + 1.0f, hippt::FLOAT_MAX));

		float3_t half_vector_unormalized = spec_data.wi + world_to_local_frame(spec_data.T, spec_data.B, shading_normal, light_lobe.axis);
		float3_t half_vector			 = half_vector_unormalized / hippt::max(hippt::length(half_vector_unormalized), hippt::FLOAT_MIN);
		float pdf						 = SGGX_reflection_PDF(spec_data.wi, half_vector, filtered_roughness_matrix);

		float3_t dominant_normal =
			local_to_world_frame(spec_data.T, spec_data.B, shading_normal, GGX_dominant_visible_normal(spec_data.wi, make_float2(alpha_x, alpha_y)));
		float3_t reflection_vector = reflect_ray(view_direction, dominant_normal) * spec_data.reflection_sharpness;

		float3_t product_vector	   = reflection_vector + light_lobe.axis * light_lobe.sharpness;
		float product_sharpness	   = hippt::length(product_vector);
		float3_t product_direction = product_vector / hippt::max(product_sharpness, hippt::FLOAT_MIN);
		float visibility		   = VMF_hemispherical_integral(hippt::dot(product_direction, shading_normal), product_sharpness);

		specular_illumination = amplitude * visibility * pdf * SG_integral(light_lobe.sharpness);
	}
#endif

	return emissive * (diffuse_illumination + specular * specular_illumination);
}

HIPRT_DEVICE float light_tree_sg_node_importance(const LightTreeSGNodeDevice& node,
												 const SGSpecularImportanceData& spec_data,
												 float3_t shading_point,
												 float3_t view_direction,
												 float3_t shading_normal,
												 float specular,
												 float alpha_x,
												 float alpha_y,
												 SGImportanceDebug* debug = nullptr)
{
	if (debug != nullptr)
		*debug = SGImportanceDebug{};

	if (node.total_power <= 0.0f)
		return 0.0f;

	if (hippt::dot(shading_normal, node.gaussian_spatial_mean - shading_point) <= 0.0f)
	{
		float3_t max_corner;
		max_corner.x = (shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x;
		max_corner.y = (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y;
		max_corner.z = (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z;

		if (hippt::dot(max_corner - shading_point, shading_normal) <= 0.0f)
			return 0.0f;
	}

	if (!DirectLightSamplingAllowBackfacingLights && light_tree_sg_node_max_emitter_cosine(node, shading_point) <= 0.0f)
		return 0.0f;

	float final_importance = 0.0f;
	for (unsigned int lobe_index = 0; lobe_index < node.spatial_lobe_count; lobe_index++)
		final_importance += light_tree_sg_evaluate_spatial_lobe(node.spatial_lobes[lobe_index], node.vmf, spec_data, shading_point, view_direction,
																shading_normal, specular, alpha_x, alpha_y);

	if (debug != nullptr)
	{
		debug->final_importance = final_importance;

		float3_t shading_to_node = node.gaussian_spatial_mean - shading_point;
		float squared_distance	 = hippt::dot(shading_to_node, shading_to_node);
		debug->squared_distance	 = squared_distance;
		debug->emitter_facing	 = hippt::dot(node.vmf.axis, -shading_to_node / hippt::max(hippt::length(shading_to_node), hippt::FLOAT_MIN));
		debug->max_corner_dot	 = hippt::dot(make_float3((shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x,
														  (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y,
														  (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z) -
												  shading_point,
											  shading_normal);
	}

	return final_importance;
}

#if LightTreeSGDoSplitting == KERNEL_OPTION_TRUE

HIPRT_DEVICE float light_tree_sg_node_raw_variance(const LightTreeSGNodeDevice& node, float3_t shading_point)
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

	return variance;
}

HIPRT_DEVICE float light_tree_sg_node_coherence(const LightTreeSGNodeDevice& node, float3_t shading_point)
{
	float variance = light_tree_sg_node_raw_variance(node, shading_point);

	return hippt::sqrt(hippt::sqrt(1.0f / (1.0f + hippt::sqrt(variance))));
}

#if LightTreeSGUseNewSplittingModel == KERNEL_OPTION_TRUE

HIPRT_DEVICE float light_tree_sg_node_best_first_split_score(unsigned int node_index,
															 const LightTreeSGNodeDevice& node,
															 const LightTreeSGNodeDevice& left_child,
															 const LightTreeSGNodeDevice& right_child,
															 const SGSpecularImportanceData& spec_data,
															 float3_t shading_point,
															 float3_t view_direction,
															 float3_t shading_normal,
															 float specular,
															 float alpha_x,
															 float alpha_y)
{
	float node_importance  = light_tree_sg_node_importance(node, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
	float left_importance  = light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
	float right_importance = light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);

	float importance_sum = left_importance + right_importance;
	float q_left		 = left_importance / importance_sum;
	float q_right		 = right_importance / importance_sum;

	float variance_left	 = hippt::max(light_tree_sg_node_raw_variance(left_child, shading_point), 0.0f);
	float variance_right = hippt::max(light_tree_sg_node_raw_variance(right_child, shading_point), 0.0f);

	float variance_reduction = variance_left * (q_right / q_left) + variance_right * (q_left / q_right);
	float unsplit_variance	 = variance_left / q_left + variance_right / q_right;

	float relative_reduction = variance_reduction / hippt::max(unsplit_variance, 1.0e-20f);

	// Use the children's local contribution estimate. The aggregate parent SG
	// importance is not additive and can differ greatly from IL + IR.
	float local_importance = importance_sum;

	float score = hippt::square(local_importance) * relative_reduction;

	return score;
}

HIPRT_DEVICE void light_tree_sg_build_best_first_split_plan(const LightTreeSGNodeDevice* nodes,
															const SGSpecularImportanceData& spec_data,
															float3_t shading_point,
															float3_t view_direction,
															float3_t shading_normal,
															float specular,
															float alpha_x,
															float alpha_y,
															unsigned int* terminal_node_indices,
															unsigned int& terminal_node_count)
{
	unsigned int candidate_node_indices[LightTreeSGSplittingMaxLightSamples] = { 0 };
	int candidate_node_count												 = 1;
	terminal_node_count														 = 0;

	while (candidate_node_count > 0)
	{
		bool performed_free_action = false;
		for (int candidate_position = 0; candidate_position < candidate_node_count; candidate_position++)
		{
			unsigned int node_index					  = candidate_node_indices[candidate_position];
			const LightTreeSGNodeDevice& current_node = nodes[node_index];
			float current_importance =
				light_tree_sg_node_importance(current_node, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
			if (current_importance <= 0.0f)
			{
				candidate_node_indices[candidate_position] = candidate_node_indices[--candidate_node_count];
				performed_free_action					   = true;

				break;
			}

			if (current_node.triangle_count != 0)
				continue;

			const LightTreeSGNodeDevice& left_child	 = nodes[current_node.left_child_index_or_first_triangle_index];
			const LightTreeSGNodeDevice& right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];
			float left_importance =
				light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
			float right_importance =
				light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);

			if (left_importance <= 0.0f && right_importance <= 0.0f)
			{
				candidate_node_indices[candidate_position] = candidate_node_indices[--candidate_node_count];
				performed_free_action					   = true;

				break;
			}

			if (left_importance <= 0.0f || right_importance <= 0.0f)
			{
				candidate_node_indices[candidate_position] =
					left_importance > 0.0f ? current_node.left_child_index_or_first_triangle_index : current_node.left_child_index_or_first_triangle_index + 1;
				performed_free_action = true;

				break;
			}
		}

		if (performed_free_action)
			continue;

		if (candidate_node_count + static_cast<int>(terminal_node_count) >= LightTreeSGSplittingMaxLightSamples)
		{
			for (int candidate_position = 0; candidate_position < candidate_node_count; candidate_position++)
				terminal_node_indices[terminal_node_count++] = candidate_node_indices[candidate_position];

			break;
		}

		int best_candidate_position	 = -1;
		float best_score			 = 0.0f;
		unsigned int best_node_index = 0;
		for (int candidate_position = 0; candidate_position < candidate_node_count; candidate_position++)
		{
			unsigned int node_index					  = candidate_node_indices[candidate_position];
			const LightTreeSGNodeDevice& current_node = nodes[node_index];
			if (current_node.triangle_count != 0)
				continue;

			const LightTreeSGNodeDevice& left_child	 = nodes[current_node.left_child_index_or_first_triangle_index];
			const LightTreeSGNodeDevice& right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];
			float score = light_tree_sg_node_best_first_split_score(node_index, current_node, left_child, right_child, spec_data, shading_point, view_direction,
																	shading_normal, specular, alpha_x, alpha_y);

			if (score > best_score || (score == best_score && (best_candidate_position < 0 || node_index < best_node_index)))
			{
				best_candidate_position = candidate_position;
				best_score				= score;
				best_node_index			= node_index;

#if LightTreeSGNewSplittingModelAlwaysSplitFirstCandidate == KERNEL_OPTION_TRUE
				break;
#endif
			}
		}

		if (best_candidate_position < 0 || best_score <= 0.0f)
		{
			for (int candidate_position = 0; candidate_position < candidate_node_count; candidate_position++)
				terminal_node_indices[terminal_node_count++] = candidate_node_indices[candidate_position];

			break;
		}

		unsigned int selected_node_index				= candidate_node_indices[best_candidate_position];
		const LightTreeSGNodeDevice& selected_node		= nodes[selected_node_index];
		candidate_node_indices[best_candidate_position] = candidate_node_indices[--candidate_node_count];
		candidate_node_indices[candidate_node_count++]	= selected_node.left_child_index_or_first_triangle_index;
		candidate_node_indices[candidate_node_count++]	= selected_node.left_child_index_or_first_triangle_index + 1;
	}
}

HIPRT_DEVICE bool light_tree_sg_plan_contains_node(const unsigned int* terminal_node_indices, unsigned int terminal_node_count, unsigned int node_index)
{
	for (unsigned int terminal_node_position = 0; terminal_node_position < terminal_node_count; terminal_node_position++)
	{
		if (terminal_node_indices[terminal_node_position] == node_index)
			return true;
	}

	return false;
}

#endif

#if LightTreeSGUseNewSplittingModel == KERNEL_OPTION_TRUE

HIPRT_DEVICE LightSampleArray<LightTreeSGSplittingMaxLightSamples> sample_one_emissive_triangle_light_tree_sg_best_first(
	const HIPRTRenderData& render_data,
	float3_t shading_point,
	float3_t view_direction,
	float3_t shading_normal,
	const SGSpecularImportanceData& spec_data,
	float specular,
	float alpha_x,
	float alpha_y,
	Xorshift32Generator& rng)
{
	const LightTreeSGNodeDevice* nodes										= render_data.light_tree_sg.nodes;
	unsigned int terminal_node_indices[LightTreeSGSplittingMaxLightSamples] = { 0 };
	unsigned int terminal_node_count										= 0;
	light_tree_sg_build_best_first_split_plan(nodes, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y,
											  terminal_node_indices, terminal_node_count);

	LightSampleArray<LightTreeSGSplittingMaxLightSamples> light_samples_out;
	unsigned int output_sample_count = 0;
	for (unsigned int terminal_node_position = 0; terminal_node_position < terminal_node_count; terminal_node_position++)
	{
		LightTreeSGNodeDevice current_node = nodes[terminal_node_indices[terminal_node_position]];
		float cumulative_probability	   = 1.0f;

		while (current_node.triangle_count == 0)
		{
			LightTreeSGNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeSGNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];
			float left_importance =
				light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
			float right_importance =
				light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
			if (left_importance <= 0.0f && right_importance <= 0.0f)
				break;

			float probability_sum  = left_importance + right_importance;
			float left_probability = left_importance / probability_sum;
			if (rng() < left_probability)
			{
				current_node = left_child;
				cumulative_probability *= left_probability;
			}
			else
			{
				current_node = right_child;
				cumulative_probability *= 1.0f - left_probability;
			}
		}

		if (current_node.triangle_count == 0 || output_sample_count >= LightTreeSGSplittingMaxLightSamples)
			continue;

		int index					= current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
		int triangle_index			= render_data.light_tree_sg.indices_array[index];
		int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

		light_samples_out[output_sample_count].emissive_triangle_global_index = emissive_triangle_index;
		light_samples_out[output_sample_count].pdf							  = cumulative_probability / current_node.triangle_count;
		output_sample_count++;
	}

	return light_samples_out;
}

#endif

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

#if LightTreeSGUseNewSplittingModel == KERNEL_OPTION_TRUE
	return sample_one_emissive_triangle_light_tree_sg_best_first(render_data, shading_point, view_direction, shading_normal, spec_data, sg_specular_weight,
																 alpha_x, alpha_y, rng);
#else
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
			float node_coherence = light_tree_sg_node_coherence(current_node, shading_point);
			if (node_coherence < render_data.light_tree_sg.settings.light_tree_sg_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold not met, exploring both branches of the tree

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
#endif
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
			float node_coherence = light_tree_sg_node_coherence(current_node, shading_point);
			if (node_coherence < render_data.light_tree_sg.settings.light_tree_sg_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold not met, exploring both branches of the tree

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

#if LightTreeSGUseNewSplittingModel == KERNEL_OPTION_TRUE

HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_sg_best_first(const HIPRTRenderData& render_data,
																	 float3_t shading_point,
																	 float3_t view_direction,
																	 float3_t shading_normal,
																	 const SGSpecularImportanceData& spec_data,
																	 float specular,
																	 float alpha_x,
																	 float alpha_y,
																	 int global_emissive_triangle_index)
{
	if (global_emissive_triangle_index == -1)
		return 0.0f;

	const LightTreeSGNodeDevice* nodes = render_data.light_tree_sg.nodes;
	LightTreeSGNodeDevice root_node	   = nodes[0];
	float root_node_importance = light_tree_sg_node_importance(root_node, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int terminal_node_indices[LightTreeSGSplittingMaxLightSamples] = { 0 };
	unsigned int terminal_node_count										= 0;
	light_tree_sg_build_best_first_split_plan(nodes, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y,
											  terminal_node_indices, terminal_node_count);

	unsigned int bit_trail			= render_data.light_tree_sg.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth		= 0;
	unsigned int current_node_index = 0;
	float cumulative_probability	= 1.0f;

	bool inside_terminal_subtree = false;
	while (nodes[current_node_index].triangle_count == 0)
	{
		const LightTreeSGNodeDevice& current_node = nodes[current_node_index];
		const LightTreeSGNodeDevice& left_child	  = nodes[current_node.left_child_index_or_first_triangle_index];
		const LightTreeSGNodeDevice& right_child  = nodes[current_node.left_child_index_or_first_triangle_index + 1];
		float left_importance = light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
		float right_importance =
			light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
		if (left_importance <= 0.0f && right_importance <= 0.0f)
			return 0.0f;

		bool go_right		   = (bit_trail & (1 << current_depth)) != 0;
		float probability_sum  = left_importance + right_importance;
		float left_probability = left_importance / probability_sum;
		unsigned int next_node_index =
			go_right ? current_node.left_child_index_or_first_triangle_index + 1 : current_node.left_child_index_or_first_triangle_index;
		float next_node_importance = go_right ? right_importance : left_importance;
		if (next_node_importance <= 0.0f)
			return 0.0f;

		if (inside_terminal_subtree || light_tree_sg_plan_contains_node(terminal_node_indices, terminal_node_count, current_node_index))
		{
			inside_terminal_subtree = true;

			cumulative_probability *= go_right ? 1.0f - left_probability : left_probability;
		}

		current_node_index = next_node_index;
		current_depth++;
	}

	return cumulative_probability / nodes[current_node_index].triangle_count;
}

#endif

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

#if LightTreeSGUseNewSplittingModel == KERNEL_OPTION_TRUE
	return pdf_of_emissive_triangle_light_tree_sg_best_first(render_data, shading_point, view_direction, shading_normal, spec_data, sg_specular_weight, alpha_x,
															 alpha_y, global_emissive_triangle_index);
#else
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
			float node_coherence = light_tree_sg_node_coherence(current_node, shading_point);
			node_split			 = node_coherence < render_data.light_tree_sg.settings.light_tree_sg_splitting_variance && current_node.triangle_count == 0;
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
#endif
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
	const bool debug				   = false; // light_tree_debug_pixel();

	if (debug)
		printf("[LT-BEGIN] algo=SG P=(%.9g,%.9g,%.9g) Ns=(%.9g,%.9g,%.9g) Ng=(%.9g,%.9g,%.9g)\n", shading_point.x, shading_point.y, shading_point.z,
			   shading_normal.x, shading_normal.y, shading_normal.z, geometric_normal.x, geometric_normal.y, geometric_normal.z);

	unsigned int current_node_index = 0;
	unsigned int depth				= 0;

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
	while (nodes[current_node_index].triangle_count == 0)
	{
		const LightTreeSGNodeDevice& current_node = nodes[current_node_index];
		const unsigned int left_index			  = current_node.left_child_index_or_first_triangle_index;
		const unsigned int right_index			  = left_index + 1;
		const LightTreeSGNodeDevice& left_child	  = nodes[left_index];
		const LightTreeSGNodeDevice& right_child  = nodes[right_index];
		SGImportanceDebug left_debug{};
		SGImportanceDebug right_debug{};

		float left_importance = light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x,
															  alpha_y, debug ? &left_debug : nullptr);
		float right_importance = light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight,
															   alpha_x, alpha_y, debug ? &right_debug : nullptr);
		if (debug)
		{
			printf("[SG-NODE] depth=%u side=L node=%u power=%.9g count=%u center=(%.9g,%.9g,%.9g) radius=%.9g vmf_axis=(%.9g,%.9g,%.9g) "
				   "orientation_axis=(%.9g,%.9g,%.9g) vmf_kappa=%.9g "
				   "dist2=%.9g max_corner_dot=%.9g emitter_facing=%.9g effective_spatial_variance=%.9g clamp_var=%.9g final_var=%.9g emissive=%.9g "
				   "light_kappa=%.9g "
				   "product_log_amp=%.9g product_kappa=%.9g product_cos=%.9g amplitude=%.9g diffuse_integral=%.9g importance=%.9g\n",
				   depth, left_index, left_child.total_power, left_child.total_emitter_count, left_child.gaussian_spatial_mean.x,
				   left_child.gaussian_spatial_mean.y, left_child.gaussian_spatial_mean.z, left_child.bounding_sphere_radius, left_child.vmf.axis.x,
				   left_child.vmf.axis.y, left_child.vmf.axis.z, left_child.orientation_axis.x, left_child.orientation_axis.y, left_child.orientation_axis.z,
				   left_child.vmf.sharpness, left_debug.squared_distance, left_debug.max_corner_dot, left_debug.emitter_facing,
				   left_debug.effective_spatial_variance, left_debug.sharpness_clamp_variance, left_debug.final_variance, left_debug.emissive,
				   left_debug.light_sharpness, left_debug.product_log_amplitude, left_debug.product_sharpness, left_debug.product_cosine, left_debug.amplitude,
				   left_debug.diffuse_integral, left_debug.final_importance);
			printf("[SG-NODE] depth=%u side=R node=%u power=%.9g count=%u center=(%.9g,%.9g,%.9g) radius=%.9g vmf_axis=(%.9g,%.9g,%.9g) "
				   "orientation_axis=(%.9g,%.9g,%.9g) vmf_kappa=%.9g "
				   "dist2=%.9g max_corner_dot=%.9g emitter_facing=%.9g effective_spatial_variance=%.9g clamp_var=%.9g final_var=%.9g emissive=%.9g "
				   "light_kappa=%.9g "
				   "product_log_amp=%.9g product_kappa=%.9g product_cos=%.9g amplitude=%.9g diffuse_integral=%.9g importance=%.9g\n",
				   depth, right_index, right_child.total_power, right_child.total_emitter_count, right_child.gaussian_spatial_mean.x,
				   right_child.gaussian_spatial_mean.y, right_child.gaussian_spatial_mean.z, right_child.bounding_sphere_radius, right_child.vmf.axis.x,
				   right_child.vmf.axis.y, right_child.vmf.axis.z, right_child.orientation_axis.x, right_child.orientation_axis.y,
				   right_child.orientation_axis.z, right_child.vmf.sharpness, right_debug.squared_distance, right_debug.max_corner_dot,
				   right_debug.emitter_facing, right_debug.effective_spatial_variance, right_debug.sharpness_clamp_variance, right_debug.final_variance,
				   right_debug.emissive, right_debug.light_sharpness, right_debug.product_log_amplitude, right_debug.product_sharpness,
				   right_debug.product_cosine, right_debug.amplitude, right_debug.diffuse_integral, right_debug.final_importance);
		}
		if (left_importance == 0.0f && right_importance == 0.0f)
		{
			if (debug)
				printf("[LT-ERROR] algo=SG depth=%u node=%u reason=both_children_zero\n", depth, current_node_index);

			return LightSampleArray<1>{ LightSampleInformation() };
		}

		float p_left			= left_importance / (left_importance + right_importance);
		float cumulative_before = cumulative_probability;
		float u					= rng();
		bool choose_left		= u < p_left;
		if (debug)
			printf("[LT-STEP] algo=SG depth=%u node=%u left=%u right=%u triangles=(%u,%u) emitters=(%u,%u) I=(%.9g,%.9g) pL=%.9g u=%.9g choice=%c "
				   "cum_before=%.9g cum_after=%.9g\n",
				   depth, current_node_index, left_index, right_index, left_child.triangle_count, right_child.triangle_count, left_child.total_emitter_count,
				   right_child.total_emitter_count, left_importance, right_importance, p_left, u, choose_left ? 'L' : 'R', cumulative_before,
				   cumulative_before * (choose_left ? p_left : 1.0f - p_left));

		if (choose_left)
		{
			current_node_index = left_index;

			cumulative_probability *= p_left;
		}
		else
		{
			current_node_index = right_index;

			cumulative_probability *= 1.0f - p_left;
		}

		depth++;
	}

	const LightTreeSGNodeDevice& current_node = nodes[current_node_index];
	int index								  = current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
	int triangle_index						  = render_data.light_tree_sg.indices_array[index];
	int emissive_triangle_index				  = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = emissive_triangle_index;
	light_sample.pdf							= cumulative_probability * (1.0f / current_node.triangle_count); // PDF of sampling that triangle in that node
	if (debug)
		printf("[LT-END] algo=SG depth=%u leaf=%u triangle_count=%u local_slot=%d triangle_index=%d emissive_global=%d cumulative=%.9g final_pdf=%.9g\n", depth,
			   current_node_index, current_node.triangle_count, index, triangle_index, emissive_triangle_index, cumulative_probability, light_sample.pdf);

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
	const bool debug				   = light_tree_debug_pixel();
	constexpr int fixed_target_light   = 7612794;
	if (debug)
		global_emissive_triangle_index = fixed_target_light;

	LightTreeSGNodeDevice current_node = nodes[0];
	unsigned int current_node_index	   = 0;

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

	unsigned int bit_trail	   = render_data.light_tree_sg.bit_trails[global_emissive_triangle_index];
	unsigned int current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		unsigned int left_index			  = current_node.left_child_index_or_first_triangle_index;
		unsigned int right_index		  = left_index + 1;
		LightTreeSGNodeDevice left_child  = nodes[left_index];
		LightTreeSGNodeDevice right_child = nodes[right_index];
		SGImportanceDebug left_debug{};
		SGImportanceDebug right_debug{};

		float left_importance = light_tree_sg_node_importance(left_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight, alpha_x,
															  alpha_y, debug ? &left_debug : nullptr);
		float right_importance = light_tree_sg_node_importance(right_child, spec_data, shading_point, view_direction, shading_normal, sg_specular_weight,
															   alpha_x, alpha_y, debug ? &right_debug : nullptr);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return 0.0f;

		float p_left					= left_importance / (left_importance + right_importance);
		bool target_is_left				= !(bit_trail & (1 << current_depth));
		float target_branch_probability = target_is_left ? p_left : 1.0f - p_left;
		cumulative_probability *= target_branch_probability;

		if (debug)
			printf("[SG PDF REPLAY %d] depth=%u node=%u target_side=%c left_importance=%.9g right_importance=%.9g target_branch_probability=%.9g "
				   "cumulative_target_probability=%.9g left_spatial_mean=(%.9g,%.9g,%.9g) right_spatial_mean=(%.9g,%.9g,%.9g) "
				   "left_effective_spatial_variance=%.9g right_effective_spatial_variance=%.9g left_vmf_sharpness=%.9g right_vmf_sharpness=%.9g\n",
				   fixed_target_light, current_depth, current_node_index, target_is_left ? 'L' : 'R', left_importance, right_importance,
				   target_branch_probability, cumulative_probability, left_child.gaussian_spatial_mean.x, left_child.gaussian_spatial_mean.y,
				   left_child.gaussian_spatial_mean.z, right_child.gaussian_spatial_mean.x, right_child.gaussian_spatial_mean.y,
				   right_child.gaussian_spatial_mean.z, left_debug.effective_spatial_variance, right_debug.effective_spatial_variance, left_child.vmf.sharpness,
				   right_child.vmf.sharpness);

		if (target_is_left)
		{
			// If the bit is not set we're going to the left
			current_node	   = left_child;
			current_node_index = left_index;
		}
		else
		{
			current_node	   = right_child;
			current_node_index = right_index;
		}

		current_depth++;
	}

	// Probability of going down the tree + probability of sampling that triangle in the node
	return cumulative_probability * 1.0f / (current_node.triangle_count);
}

#endif

#endif
