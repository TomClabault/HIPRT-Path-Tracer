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

HIPRT_DEVICE float light_tree_node_importance(const LightTreeNodeDevice& node, float3 shading_point, float3 surface_normal)
{
	float3 node_center = (node.bounds_max + node.bounds_min) * 0.5f;
	// Using a minimum for the distance squared to avoid large errors if a point is very close to the center
	// of the node for example
	float distance_squared = hippt::max(hippt::length2(node_center - shading_point), hippt::length(node.bounds_max - node.bounds_min));


	// cos_theta_u is going to be computed as a conservative bound using a sphere
	// bounding the node
	float3 half_extents = (node.bounds_max - node.bounds_min) * 0.5f;
	float sphere_radius = hippt::length(half_extents);

	float theta_u = asin(hippt::min(1.0f, sphere_radius / sqrtf(distance_squared)));
	float theta_i = acos(hippt::dot(surface_normal, hippt::normalize(node_center - shading_point)));
	float theta_i_prime = hippt::max(0.0f, theta_i - theta_u);

	float theta = acos(hippt::dot(node.axis, hippt::normalize(shading_point - node_center)));
	float theta_prime = hippt::max(0.0f, theta - node.theta_o - theta_u);

#if DirectLightSamplingAllowBackfacingLights == KERNEL_OPTION_TRUE
	float cos_theta_prime = hippt::abs(cos(theta_prime));
#else
	float cos_theta_prime = hippt::max(0.0f, cos(theta_prime));
#endif
	
	return hippt::abs(cos(theta_i_prime)) * node.total_power.luminance() / distance_squared * cos_theta_prime;
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

	int triangle_index = render_data.buffers.light_tree.indices_array[current_node.first_triangle_index + rng.random_index(current_node.triangle_count)];
	int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample = sample_point_on_generic_triangle_and_fill_light_sample_information(render_data, emissive_triangle_index, rng);
	light_sample.area_measure_pdf *= cumulative_probability;
	light_sample.area_measure_pdf *= 1.0f / current_node.triangle_count; // Sampling that triangle in that node

	return light_sample;
}

#endif
