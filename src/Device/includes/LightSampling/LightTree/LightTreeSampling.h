/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_SAMPLING_H

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

HIPRT_DEVICE float light_tree_node_importance(const LightTreeNodeDevice& node, float3 shading_point, float3 surface_normal)
{
	// If the whole node is behind the surface, quick exit
	//
	// Not doing it this way and relying on the bounding sphere of the node as done
	// later isn't enough sometimes so this helps a lot in cases where the bounding
	// sphere is too conservative
	if (hippt::dot(make_float3(node.bounds_min.x, node.bounds_min.y, node.bounds_min.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_min.y, node.bounds_min.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_min.x, node.bounds_max.y, node.bounds_min.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_max.y, node.bounds_min.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_min.x, node.bounds_min.y, node.bounds_max.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_min.y, node.bounds_max.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_min.x, node.bounds_max.y, node.bounds_max.z) - shading_point, surface_normal) <= 0.0f &&
		hippt::dot(make_float3(node.bounds_max.x, node.bounds_max.y, node.bounds_max.z) - shading_point, surface_normal) <= 0.0f)
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
	float theta_i = acos(hippt::dot(surface_normal, hippt::normalize(node_center - shading_point)));
	float theta_i_prime = hippt::max(0.0f, theta_i - theta_u);

	float theta = acos(hippt::dot(node.axis, hippt::normalize(shading_point - node_center)));
	float theta_prime = hippt::max(0.0f, theta - node.theta_o - theta_u);

#if DirectLightSamplingAllowBackfacingLights == KERNEL_OPTION_TRUE
	float cos_theta_prime = hippt::abs(cos(theta_prime));
#else
	float cos_theta_prime = hippt::max(0.0f, cosf(theta_prime));
#endif
	
	return hippt::abs(cos(theta_i_prime)) * node.total_power.luminance() / distance_to_center_2 * cos_theta_prime;
}

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree(const HIPRTRenderData& render_data, float3 shading_point, float3 surface_normal, Xorshift32Generator& rng)
{
	const LightTreeNodeDevice* nodes = render_data.buffers.light_tree.nodes;

	LightTreeNodeDevice current_node = nodes[0];

	float root_node_importance = light_tree_node_importance(current_node, shading_point, surface_normal);
	if (root_node_importance <= 0)
		return LightSampleInformation();

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeNodeDevice left_child = nodes[current_node.left_child_index];
		LightTreeNodeDevice right_child = nodes[current_node.left_child_index + 1];

		float left_importance = light_tree_node_importance(left_child, shading_point, surface_normal);
		float right_importance = light_tree_node_importance(right_child, shading_point, surface_normal);

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
