/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_SAMPLING_H

#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/TriangleSampling.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE bool point_inside_AABB(float3 aabb_min, float3 aabb_max, float3 point)
{
	return (point.x <= aabb_max.x && point.x >= aabb_min.x) &&
		(point.y <= aabb_max.y && point.y >= aabb_min.y) &&
		(point.z <= aabb_max.z && point.z >= aabb_min.z);
}

HIPRT_DEVICE float subtended_angle_aabb_to_point_average_corners(float3 aabb_min, float3 aabb_max, float3 point)
{
	if (point_inside_AABB(aabb_min, aabb_max, point))
		return M_PI;

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

HIPRT_DEVICE float light_tree_node_importance(const LightTreeNodeDevice& node, float3 shading_point, float3 shading_normal)
{
	if (node.is_invalid())
		return 0.0f;

	// If the whole node is behind the surface, quick exit
	//
	// Not doing it this way and relying on the bounding sphere of the node as done
	// later isn't enough sometimes so this helps a lot in cases where the bounding
	// sphere is too conservative
	if (hippt::dot(make_float3(node.bounds_min.x, node.bounds_min.y, node.bounds_min.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_min.y, node.bounds_min.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_min.x, node.bounds_max.y, node.bounds_min.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_max.y, node.bounds_min.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_min.x, node.bounds_min.y, node.bounds_max.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_min.y, node.bounds_max.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_min.x, node.bounds_max.y, node.bounds_max.z) - shading_point, shading_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_max.y, node.bounds_max.z) - shading_point, shading_normal) <= 0.0f)
		return 0.0f;

	float3 node_center = (node.bounds_max + node.bounds_min) * 0.5f;
	// Using a minimum for the distance squared to avoid large errors if a point is very close to the center
	// of the node for example
	float distance_to_center_2 = hippt::length2(node_center - shading_point);
	distance_to_center_2 = hippt::max(distance_to_center_2, hippt::length(node.bounds_max - node.bounds_min));

	// cos_theta_u is going to be computed as a conservative bound using a sphere
	// bounding the node
	float3 half_extents = (node.bounds_max - node.bounds_min) * 0.5f;
	float sphere_radius = hippt::length(half_extents);

	float theta_u;
	if (point_inside_AABB(node.bounds_min, node.bounds_max, shading_point))
		theta_u = M_PI;
	else
		theta_u = asin(hippt::min(1.0f, sphere_radius / sqrtf(hippt::length2(node_center - shading_point))));
	//float theta_u = subtended_angle_aabb_to_point_average_corners(node.bounds_min, node.bounds_max, shading_point);
	float theta_i = acos(hippt::clamp(0.0f, 1.0f, hippt::dot(shading_normal, hippt::normalize(node_center - shading_point))));
	float theta_i_prime = hippt::max(0.0f, theta_i - theta_u);

	float theta = acos(hippt::clamp(0.0f, 1.0f, hippt::dot(node.axis, hippt::normalize(shading_point - node_center))));
	float theta_prime = hippt::max(0.0f, theta - node.theta_o - theta_u);

#if DirectLightSamplingAllowBackfacingLights == KERNEL_OPTION_TRUE
	float cos_theta_prime = hippt::abs(cos(theta_prime));
#else
	float cos_theta_prime = hippt::max(0.0f, cosf(theta_prime));
#endif
	
	return hippt::abs(cos(theta_i_prime)) * node.total_power.luminance() / distance_to_center_2 * cos_theta_prime;
}

HIPRT_DEVICE float light_tree_node_variance(const LightTreeNodeDevice& node, float3 shading_point)
{
	float3 node_center = (node.bounds_max + node.bounds_min) * 0.5f;
	float3 half_extents = (node.bounds_max - node.bounds_min) * 0.5f;
	float bounding_sphere_radius = hippt::length(half_extents);

	// Compute a and b for the geometric mean and variance
	//float a = hippt::max(hippt::length(shading_point - node_center) - bounding_sphere_radius, hippt::length(node.bounds_max - node.bounds_min));
	float a = hippt::max(hippt::length(shading_point - node_center) - bounding_sphere_radius, 1.0e-3f);
	float b = hippt::length(shading_point - node_center) + bounding_sphere_radius;

	float a3 = hippt::pow_3(a);
	float b3 = hippt::pow_3(b);

	float mean_geometric = 1.0f / (a * b);
	float variance_geometric = (b3 - a3) / (3.0f * (b - a) * a3 * b3) - 1.0f / (a * a * b * b);
	float variance = (node.energy_variance * variance_geometric + node.energy_variance * hippt::square(mean_geometric) + hippt::square(node.energy_average) * variance_geometric) * hippt::square(node.total_emitter_count);

	return sqrtf(sqrtf(1.0f / (1.0f + sqrtf(variance))));
}

#if LightTreeATSDoSplitting == KERNEL_OPTION_TRUE

#define ATS_LIGHT_TREE_SPLITTING_STACK_SIZE 64

struct LightTreeATSWRSReservoir
{
	HIPRT_DEVICE float compute_light_sample_weight(const HIPRTRenderData& render_data, const LightSampleInformation& light_sample, 
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
			bool in_shadow = evaluate_shadow_ray_nee_plus_plus(const_cast<HIPRTRenderData&>(render_data), shadow_ray, distance_to_light, last_hit_primitive_index, nee_plus_plus_context, rng, ray_payload.bounce);
#else
			bool in_shadow = false;
#endif

			if (!in_shadow)
			{
				float bsdf_pdf;

				BSDFIncidentLightInfo incident_light_info = light_sample.incident_light_info;
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

						return weight;
					}
				}
			}
		}

		return 0.0f;
	}

	HIPRT_DEVICE bool stream_sample(const HIPRTRenderData& render_data, const LightSampleInformation& light_sample, 
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
			selected_light_sample = light_sample;

			return true;
		}

		return false;
	}

	float weight_sum = 0.0f;

	float selected_sample_weight = 0.0f;
	LightSampleInformation selected_light_sample;
};

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree(const HIPRTRenderData& render_data, 
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal, 
	int last_hit_primitive_index, RayPayload& ray_payload,
	Xorshift32Generator& rng)
{
	const LightTreeNodeDevice* nodes = render_data.buffers.light_tree.nodes;

	int stack_pointer = 0;
	unsigned int node_index_stack[ATS_LIGHT_TREE_SPLITTING_STACK_SIZE] = { 0 };
	unsigned int light_samples_counter = 1;

	int node_indices_to_be_traversed_sp = 0;
	int node_indices_to_be_traversed[LightTreeATSSplittingMaxLightSamples] = { -1 };

	while (light_samples_counter < LightTreeATSSplittingMaxLightSamples && stack_pointer >= 0)
	{
		unsigned int node_index = node_index_stack[stack_pointer--];
		LightTreeNodeDevice current_node = nodes[node_index];

		light_samples_counter--;

		float node_importance = light_tree_node_importance(current_node, shading_point, shading_normal);
		if (node_importance > 0.0f)
		{
			float node_variance = light_tree_node_variance(current_node, shading_point);
			if (node_variance < render_data.light_tree_ats_settings.light_tree_ats_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold exceeded, exploring both branches of the tree

				float node_importance_left = light_tree_node_importance(nodes[current_node.left_child_index], shading_point, shading_normal);
				float node_importance_right = light_tree_node_importance(nodes[current_node.right_child_index], shading_point, shading_normal);

				if (node_importance_left > node_importance_right)
				{
					// We're going to want to explore the left child first

					// So inserting the right child first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.right_child_index;
						light_samples_counter++;
					}

					// And then the left child such that the left child is popped first
					// and explored first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index;
						light_samples_counter++;
					}
				}
				else
				{
					// We're going to want to explore the right child first

					// So inserting the left child first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index;
						light_samples_counter++;
					}

					// And then the right child such that the right child is popped first
					// and explored first
					if (stack_pointer < ATS_LIGHT_TREE_SPLITTING_STACK_SIZE - 1 && node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.right_child_index;
						light_samples_counter++;
					}
				}
			}
			else
			{
				light_samples_counter++;

				node_indices_to_be_traversed[node_indices_to_be_traversed_sp++] = node_index;
				// Variance threshold cool, picking a light from this sub tree

				// Keep the node in the stack or something
				// ......
			}
		}
	}

	LightTreeATSWRSReservoir wrs;

	// Stream candidates that were left in the stack for splitting
	while (stack_pointer >= 0)
	{
		unsigned int node_index = node_index_stack[stack_pointer--];
		LightTreeNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeNodeDevice left_child = nodes[current_node.left_child_index];
			LightTreeNodeDevice right_child = nodes[current_node.right_child_index];

			float left_importance = light_tree_node_importance(left_child, shading_point, shading_normal);
			float right_importance = light_tree_node_importance(right_child, shading_point, shading_normal);

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
			int index = current_node.first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index = render_data.buffers.light_tree.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
			light_sample.area_measure_pdf *= cumulative_probability;
			light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

			wrs.stream_sample(render_data, light_sample,
				shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				rng);
		}

		// We have found a good node in this tree
	}

	// Also stream candidates that we're deemed as not needing splitting
	while (node_indices_to_be_traversed_sp > 0)
	{
		unsigned int node_index = node_indices_to_be_traversed[--node_indices_to_be_traversed_sp];
		LightTreeNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeNodeDevice left_child = nodes[current_node.left_child_index];
			LightTreeNodeDevice right_child = nodes[current_node.right_child_index];

			float left_importance = light_tree_node_importance(left_child, shading_point, shading_normal);
			float right_importance = light_tree_node_importance(right_child, shading_point, shading_normal);

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
			int index = current_node.first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index = render_data.buffers.light_tree.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
			light_sample.area_measure_pdf *= cumulative_probability;
			light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

			wrs.stream_sample(render_data, light_sample,
				shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				rng);
		}

		// We have found a good node in this tree
	}

	LightSampleInformation& final_light_sample = wrs.selected_light_sample;
	if (final_light_sample.emissive_triangle_global_index == -1)
		return LightSampleInformation();

	float light_sample_wrs_PDF = wrs.selected_sample_weight / wrs.weight_sum;
	final_light_sample.area_measure_pdf *= light_sample_wrs_PDF;

	return final_light_sample;
}
#else
HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree(const HIPRTRenderData& render_data,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal,
	int last_hit_primitive_index, RayPayload& ray_payload,
	Xorshift32Generator& rng)
{
	const LightTreeNodeDevice* nodes = render_data.buffers.light_tree.nodes;

	LightTreeNodeDevice current_node = nodes[0];

	float root_node_importance = light_tree_node_importance(current_node, shading_point, shading_normal);
	if (root_node_importance <= 0)
		return LightSampleInformation();

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeNodeDevice left_child = nodes[current_node.left_child_index];
		LightTreeNodeDevice right_child = nodes[current_node.right_child_index];

		float left_importance = light_tree_node_importance(left_child, shading_point, shading_normal);
		float right_importance = light_tree_node_importance(right_child, shading_point, shading_normal);
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

	int index = current_node.first_triangle_index + rng.random_index(current_node.triangle_count);
	int triangle_index = render_data.buffers.light_tree.indices_array[index];
	int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
	light_sample.area_measure_pdf *= cumulative_probability;
	light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

	return light_sample;
}
#endif

HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree(const HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal, int global_emissive_triangle_index)
{
	const LightTreeNodeDevice* nodes = render_data.buffers.light_tree.nodes;

	LightTreeNodeDevice current_node = nodes[0];

	float root_node_importance = light_tree_node_importance(current_node, shading_point, shading_normal);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail = render_data.buffers.light_tree.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeNodeDevice left_child = nodes[current_node.left_child_index];
		LightTreeNodeDevice right_child = nodes[current_node.right_child_index];

		float left_importance = light_tree_node_importance(left_child, shading_point, shading_normal);
		float right_importance = light_tree_node_importance(right_child, shading_point, shading_normal);
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
