/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_ATS_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_ATS_SAMPLING_H

#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/TriangleSampling.h"

#include "HostDeviceCommon/KernelOptions/LightTreeATSOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE HIPRT_INLINE bool point_inside_AABB(float3 aabb_min, float3 aabb_max, float3 point)
{
	return (point.x <= aabb_max.x && point.x >= aabb_min.x) &&
		(point.y <= aabb_max.y && point.y >= aabb_min.y) &&
		(point.z <= aabb_max.z && point.z >= aabb_min.z);
}

HIPRT_DEVICE float subtended_angle_aabb_to_point_average_corners(float3 aabb_min, float3 aabb_max, float3 point)
{
	if (point_inside_AABB(aabb_min, aabb_max, point))
		return hippt::M_Pi;

	// Compute the average vector to each of the bounding box corners to get the direction of
	// the bounding cone
	float3 direction_to_corners_sum = make_float3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < 8; ++i)
	{
		float3 corner = make_float3((i & 1) ? aabb_min.x : aabb_max.x, (i & 2) ? aabb_min.y : aabb_max.y, (i & 4) ? aabb_min.z : aabb_max.z);
		direction_to_corners_sum += hippt::normalize(corner - point);
	}

	float3 cone_direction = hippt::normalize(direction_to_corners_sum);

	// Now that we have the cone direction, compute the angle that cone with that forms 
	// with each corner of the bounds and keep the largest angle (which is the min cos theta)
	// 
	// Compute the cosine of the maximum angle between a corner and the
	// average vector.
	float cos_theta = 1.0f;
	for (int i = 0; i < 8; ++i)
	{
		float3 corner = make_float3((i & 1) ? aabb_min.x : aabb_max.x, (i & 2) ? aabb_min.y : aabb_max.y, (i & 4) ? aabb_min.z : aabb_max.z);
		cos_theta = hippt::min(cos_theta, hippt::dot(hippt::normalize(corner - point), cone_direction));
	}

	return acos(cos_theta);
}

template <bool UseOrientation>
HIPRT_DEVICE float light_tree_ats_node_importance(const LightTreeATSNodeDevice& node, float3 shading_point, float3 shading_normal)
{
	if (node.is_invalid())
		return 0.0f;

	// If the whole node is behind the surface, quick exit (if even the corner that maximizes
	// the dot product yields a dot product negative, then every corners are going to be behind
	// the surface)
	//
	// Not doing it this way and relying on the bounding sphere of the node as done
	// later isn't enough sometimes so this helps a lot in cases where the bounding
	// sphere is too conservative
	if constexpr (UseOrientation)
	{
		float3 max_corner;
		max_corner.x = (shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x;
		max_corner.y = (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y;
		max_corner.z = (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z;

		if (hippt::dot(max_corner - shading_point, shading_normal) <= 0.0f)
			return 0.0f;
	}

	float3 node_center = (node.bounds_max + node.bounds_min) * 0.5f;
	float3 to_center = node_center - shading_point;
	float dist_to_center = hippt::length(to_center);
	float3 to_center_normalized = to_center / dist_to_center;
	float3 node_diag = node.bounds_max - node.bounds_min;
	float half_diag_length = hippt::length(node_diag) * 0.5f;
	// Using a minimum for the distance squared to avoid large errors if a point is very close to the center
	// of the node for example
	float distance_to_center_2 = hippt::max(hippt::length2(node_center - shading_point), half_diag_length * 2.0f);

	float sphere_radius = half_diag_length;
	bool inside_aabb = point_inside_AABB(node.bounds_min, node.bounds_max, shading_point);
	float sin_theta_u = 0.0f;
	float cos_theta_u = 1.0f;
	if (inside_aabb)
	{
		// If the shading point is inside the bounds, treat theta_u as PI
		sin_theta_u = 0.0f;
		cos_theta_u = -1.0f;
	}
	else
	{
		float ratio = sphere_radius / dist_to_center;
		if (ratio >= 1.0f)
		{
			sin_theta_u = 1.0f;
			cos_theta_u = 0.0f;
		}
		else
		{
			sin_theta_u = ratio;
			cos_theta_u = hippt::sqrt(hippt::max(0.0f, 1.0f - sin_theta_u * sin_theta_u));
		}
	}

	float cos_theta_i_prime;
	if (inside_aabb)
		// Inside:
		// => theta_u == PI 
		// => theta_i_prime = max(0, theta_i - PI) = 0 
		// => cos = 1
		cos_theta_i_prime = 1.0f;
	else
	{
		// theta_i <= theta_u <=> cos(theta_i) >= cos(theta_u)
		float cos_theta_i = hippt::dot(shading_normal, to_center_normalized);
		if (cos_theta_i >= cos_theta_u)
			cos_theta_i_prime = 1.0f;
		else
		{
			// cos(theta_i - theta_u) = cos(theta_i) * cos(theta_u) + sin(theta_i) * sin(theta_u)
			float sin_theta_i = hippt::sqrt(hippt::max(0.0f, 1.0f - cos_theta_i * cos_theta_i));
			cos_theta_i_prime = cos_theta_i * cos_theta_u + sin_theta_i * sin_theta_u;
		}
	}

	float cos_theta = hippt::clamp(hippt::dot(node.axis, -to_center_normalized), 0.0f, 1.0f);
	float sin_theta = hippt::sqrt(hippt::max(0.0f, 1.0f - cos_theta * cos_theta));

	// For T = node.theta_o + theta_u
	// Compute cos_T and sin_T
	float cos_T = node.cos_theta_o * cos_theta_u - node.sin_theta_o * sin_theta_u;
	float sin_T = node.sin_theta_o * cos_theta_u + node.cos_theta_o * sin_theta_u;

	float cos_theta_prime;
	if (sin_T <= 0.0f)
		// T >= pi (which is sin_T <= 0.0f)
		// -> theta (in [0,pi/2]) <= T
		// always -> theta' == 0 -> cos(theta') == 1
		cos_theta_prime = 1.0f;
	else
	{
		// Now T in (0, pi) so cos is monotonic and we can compare cosines
		if (cos_theta >= cos_T)
			cos_theta_prime = 1.0f;
		else
		{
			// cos(theta - T) = cos(theta) * cos(T) + sin(theta) * sin(T)
			cos_theta_prime = cos_theta * cos_T + sin_theta * sin_T;

#if DirectLightSamplingAllowBackfacingLights
			cos_theta_prime = hippt::abs(cos_theta_prime);
#else
			cos_theta_prime = hippt::max(0.0f, cos_theta_prime);
#endif
		}
	}

	if constexpr (UseOrientation)
		return cos_theta_i_prime * node.total_power_luminance / distance_to_center_2 * cos_theta_prime;
	else
		return node.total_power_luminance / distance_to_center_2;
}

#if LightTreeATSDoSplitting == KERNEL_OPTION_TRUE

#define ATS_LIGHT_TREE_SPLITTING_STACK_SIZE 64

HIPRT_DEVICE float light_tree_ats_node_variance(const LightTreeATSNodeDevice& node, float3 shading_point)
{
	float3 node_center = (node.bounds_max + node.bounds_min) * 0.5f;
	float3 half_extents = (node.bounds_max - node.bounds_min) * 0.5f;
	float bounding_sphere_radius = hippt::length(half_extents);

	// Compute a and b for the geometric mean and variance
	float a = hippt::max(hippt::length(shading_point - node_center) - bounding_sphere_radius, 1.0e-3f);
	float b = hippt::length(shading_point - node_center) + bounding_sphere_radius;

	float a3 = hippt::pow_3(a);
	float b3 = hippt::pow_3(b);

	float mean_geometric = 1.0f / (a * b);
	float variance_geometric = (b3 - a3) / (3.0f * (b - a) * a3 * b3) - 1.0f / (a * a * b * b);
	float variance = (node.energy_variance * variance_geometric + node.energy_variance * hippt::square(mean_geometric) + hippt::square(node.get_energy_average()) * variance_geometric) * hippt::square(node.total_emitter_count);

	return hippt::sqrt(hippt::sqrt(1.0f / (1.0f + hippt::sqrt(variance))));
}

struct LightTreeATSWRSReservoir
{
	HIPRT_DEVICE float compute_light_sample_weight(const HIPRTRenderData& render_data, const LightSamplePointInformation& light_sample, 
		float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal,
		int last_hit_primitive_index, RayPayload& ray_payload,
		Xorshift32Generator& rng)
	{
		float3 shadow_ray_origin = shading_point;
		float3 shadow_ray_direction = light_sample.point_on_light - shadow_ray_origin;
		float distance_to_light = hippt::length(shadow_ray_direction);
		float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

		hiprtRay shadow_ray;
		shadow_ray.origin = shadow_ray_origin;
		shadow_ray.direction = shadow_ray_direction_normalized;

		// abs() here to allow backfacing light sources
		float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);
		if (dot_light_source > 0.0f)
		{
			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point = shadow_ray_origin;

#if LightTreeATSSplittingIncludeVisibility == KERNEL_OPTION_TRUE
#if LightTreeATSSplittingDoNEEPlusPlusVisibility == KERNEL_OPTION_TRUE && DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
			bool in_shadow = false;
#else
			bool in_shadow = evaluate_shadow_ray_nee_plus_plus(const_cast<HIPRTRenderData&>(render_data), shadow_ray, distance_to_light, last_hit_primitive_index, nee_plus_plus_context, rng, ray_payload.bounce);
#endif
#else
			bool in_shadow = false;
#endif

			if (!in_shadow)
			{
				float bsdf_pdf;

				BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
				BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, shadow_ray.direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
				BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, shadow_ray.direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif
				ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, rng);

				if (bsdf_pdf != 0.0f)
				{
					// Conversion to solid angle from surface area measure
					float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
					if (light_sample_solid_angle_pdf > 0.0f)
					{
						float cosine_term = hippt::abs(hippt::dot(shading_normal, shadow_ray.direction));
						float weight = (light_sample.emission * cosine_term * bsdf_color / light_sample_solid_angle_pdf / nee_plus_plus_context.unoccluded_probability).luminance();

#if LightTreeATSSplittingIncludeVisibility == KERNEL_OPTION_TRUE && LightTreeATSSplittingDoNEEPlusPlusVisibility == KERNEL_OPTION_TRUE && DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
						weight *= hippt::max(0.025f, render_data.nee_plus_plus.estimate_visibility_probability(nee_plus_plus_context, render_data.current_camera));
#endif

						return weight;
					}
				}
			}
		}

		return 0.0f;
	}

	HIPRT_DEVICE bool stream_sample(const HIPRTRenderData& render_data, const LightSamplePointInformation& light_sample, 
		float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal,
		int last_hit_primitive_index, RayPayload& ray_payload, 
		Xorshift32Generator& rng)
	{
		float light_sample_weight = compute_light_sample_weight(render_data, light_sample,
			shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
			rng);

		weight_sum += light_sample_weight;

		if (rng() < light_sample_weight / weight_sum)
		{
			selected_sample_weight = light_sample_weight;
			selected_light_index = light_sample.emissive_triangle_global_index;

			return true;
		}

		return false;
	}

	float weight_sum = 0.0f;

	float selected_sample_weight = 0.0f;
	float selected_light_pdf = 0.0f;
	int selected_light_index = -1;
};

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal, 
	int last_hit_primitive_index, RayPayload& ray_payload,
	Xorshift32Generator& rng)
{
	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;

	int stack_pointer = 0;
	unsigned int node_index_stack[ATS_LIGHT_TREE_SPLITTING_STACK_SIZE] = { 0 };
	unsigned int light_samples_counter = 1;

	int node_indices_to_be_traversed_sp = 0;
	int node_indices_to_be_traversed[LightTreeATSSplittingMaxLightSamples] = { -1 };

	while (light_samples_counter < LightTreeATSSplittingMaxLightSamples && stack_pointer >= 0)
	{
		unsigned int node_index = node_index_stack[stack_pointer--];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		light_samples_counter--;

		float node_importance = light_tree_ats_node_importance<UseOrientation>(current_node, shading_point, shading_normal);
		if (node_importance > 0.0f)
		{
			float node_variance = light_tree_ats_node_variance(current_node, shading_point);
			if (node_variance < render_data.light_tree_ats.settings.light_tree_ats_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold exceeded, exploring both branches of the tree

				float node_importance_left = light_tree_ats_node_importance<UseOrientation>(nodes[current_node.left_child_index_or_first_triangle_index], shading_point, shading_normal);
				float node_importance_right = light_tree_ats_node_importance<UseOrientation>(nodes[current_node.left_child_index_or_first_triangle_index + 1], shading_point, shading_normal);

				if (node_importance_left > node_importance_right)
				{
					// We're going to want to explore the left child first

					// So inserting the right child first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
						light_samples_counter++;
					}

					// And then the left child such that the left child is popped first
					// and explored first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						light_samples_counter++;
					}
				}
				else
				{
					// We're going to want to explore the right child first

					// So inserting the left child first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						light_samples_counter++;
					}

					// And then the right child such that the right child is popped first
					// and explored first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_right > 0.0f)
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

	LightTreeATSWRSReservoir wrs;

	// Stream candidates that were left in the stack for splitting
	while (stack_pointer >= 0)
	{
		unsigned int node_index = node_index_stack[stack_pointer--];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeATSNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

			float left_importance = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
			float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);

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
			int index = current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index = render_data.light_tree_ats.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			LightSamplePointInformation light_sample = sample_point_on_light_and_fill_light_sample_information(render_data, 
				shading_point, view_direction, shading_normal,
				ray_payload.material,
				emissive_triangle_index, rng);
			light_sample.area_measure_pdf *= cumulative_probability;
			light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

			if (wrs.stream_sample(render_data, light_sample,
				shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				rng))
				wrs.selected_light_pdf = cumulative_probability / current_node.triangle_count;
		}

		// We have found a good node in this tree
	}

	// Also stream candidates that we're deemed as not needing splitting
	while (node_indices_to_be_traversed_sp > 0)
	{
		unsigned int node_index = node_indices_to_be_traversed[--node_indices_to_be_traversed_sp];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeATSNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

			float left_importance = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
			float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);

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

			int index = current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index = render_data.light_tree_ats.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			LightSamplePointInformation light_sample = sample_point_on_light_and_fill_light_sample_information(render_data, 
				shading_point, view_direction, shading_normal,
				ray_payload.material,
				emissive_triangle_index, rng);
			light_sample.area_measure_pdf *= cumulative_probability;
			light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

			if (wrs.stream_sample(render_data, light_sample,
				shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				rng))
				wrs.selected_light_pdf = cumulative_probability / current_node.triangle_count;
		}
	}

	if (wrs.selected_light_index == -1)
		return LightSampleInformation();

	LightSampleInformation final_light_sample;
	final_light_sample.emissive_triangle_global_index = wrs.selected_light_index;
	final_light_sample.pdf = wrs.selected_sample_weight / wrs.weight_sum;
	final_light_sample.pdf *= wrs.selected_light_pdf;

	return final_light_sample;
}

#else

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal,
	int last_hit_primitive_index, RayPayload& ray_payload,
	Xorshift32Generator& rng)
{
	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;

	LightTreeATSNodeDevice current_node = nodes[0];

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeATSNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
		float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);
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
	int triangle_index = render_data.light_tree_ats.indices_array[index];
	int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = emissive_triangle_index;
	light_sample.pdf = cumulative_probability * (1.0f / current_node.triangle_count); // Sampling that triangle in that node

	return light_sample;
}
#endif

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal, int global_emissive_triangle_index)
{
	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;

	LightTreeATSNodeDevice current_node = nodes[0];

	float root_node_importance = light_tree_ats_node_importance<UseOrientation>(current_node, shading_point, shading_normal);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail = render_data.light_tree_ats.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeATSNodeDevice left_child = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
		float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);
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
	return cumulative_probability / current_node.triangle_count;
}

#endif
