/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_ATS_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_ATS_SAMPLING_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/TriangleSampling.h"

#include "HostDeviceCommon/KernelOptions/LightTreeATSOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE HIPRT_INLINE bool point_inside_AABB(float3_t aabb_min, float3_t aabb_max, float3_t point)
{
	return (point.x <= aabb_max.x && point.x >= aabb_min.x) && (point.y <= aabb_max.y && point.y >= aabb_min.y) &&
		   (point.z <= aabb_max.z && point.z >= aabb_min.z);
}

HIPRT_DEVICE float subtended_angle_aabb_to_point_average_corners(float3_t aabb_min, float3_t aabb_max, float3_t point)
{
	if (point_inside_AABB(aabb_min, aabb_max, point))
		return hippt::M_Pi;

	// Compute the average vector to each of the bounding box corners to get the direction of
	// the bounding cone
	float3_t direction_to_corners_sum = make_float3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < 8; ++i)
	{
		float3_t corner = make_float3((i & 1) ? aabb_min.x : aabb_max.x, (i & 2) ? aabb_min.y : aabb_max.y, (i & 4) ? aabb_min.z : aabb_max.z);
		direction_to_corners_sum += hippt::normalize(corner - point);
	}

	float3_t cone_direction = hippt::normalize(direction_to_corners_sum);

	// Now that we have the cone direction, compute the angle that cone with that forms
	// with each corner of the bounds and keep the largest angle (which is the min cos theta)
	//
	// Compute the cosine of the maximum angle between a corner and the
	// average vector.
	float cos_theta = 1.0f;
	for (int i = 0; i < 8; ++i)
	{
		float3_t corner = make_float3((i & 1) ? aabb_min.x : aabb_max.x, (i & 2) ? aabb_min.y : aabb_max.y, (i & 4) ? aabb_min.z : aabb_max.z);
		cos_theta		= hippt::min(cos_theta, hippt::dot(hippt::normalize(corner - point), cone_direction));
	}

	return acos(cos_theta);
}

template <bool UseOrientation>
HIPRT_DEVICE float light_tree_ats_node_importance(const LightTreeATSNodeDevice& node, float3_t shading_point, float3_t shading_normal)
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
		float3_t max_corner;
		max_corner.x = (shading_normal.x >= 0.0f) ? node.bounds_max.x : node.bounds_min.x;
		max_corner.y = (shading_normal.y >= 0.0f) ? node.bounds_max.y : node.bounds_min.y;
		max_corner.z = (shading_normal.z >= 0.0f) ? node.bounds_max.z : node.bounds_min.z;

		float max_corner_dot = hippt::dot(max_corner - shading_point, shading_normal);
		if (max_corner_dot <= 0.0f)
			return 0.0f;
	}

	float3_t node_center		  = (node.bounds_max + node.bounds_min) * 0.5f;
	float3_t to_center			  = node_center - shading_point;
	float dist_to_center		  = hippt::length(to_center);
	float3_t to_center_normalized = to_center / dist_to_center;
	float3_t node_diag			  = node.bounds_max - node.bounds_min;
	float half_diag_length		  = hippt::length(node_diag) * 0.5f;
	// Using a minimum for the distance squared to avoid large errors if a point is very close to the center
	// of the node for example
	float distance_to_center_2 = hippt::max(hippt::length2(node_center - shading_point), half_diag_length * 2.0f);

	float sphere_radius = half_diag_length;
	bool inside_aabb	= point_inside_AABB(node.bounds_min, node.bounds_max, shading_point);
	float sin_theta_u	= 0.0f;
	float cos_theta_u	= 1.0f;
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

	const float final_importance = UseOrientation ? cos_theta_i_prime * node.total_power_luminance / distance_to_center_2 * cos_theta_prime
												  : node.total_power_luminance / distance_to_center_2;

	return final_importance;
}

#if LightTreeATSDoSplitting == KERNEL_OPTION_TRUE

#define ATS_LIGHT_TREE_SPLITTING_STACK_SIZE 2

HIPRT_DEVICE float light_tree_ats_node_coherence(const LightTreeATSNodeDevice& node, float3_t shading_point)
{
	float3_t node_center		 = (node.bounds_max + node.bounds_min) * 0.5f;
	float3_t half_extents		 = (node.bounds_max - node.bounds_min) * 0.5f;
	float bounding_sphere_radius = hippt::length(half_extents);

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

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE LightSampleArray<DirectLightSampleCount<LSS_BASE_LIGHT_TREE_ATS>()> sample_one_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data,
																															 float3_t shading_point,
																															 float3_t view_direction,
																															 float3_t shading_normal,
																															 float3_t geometric_normal,
																															 int last_hit_primitive_index,
																															 RayPayload& ray_payload,
																															 Xorshift32Generator& rng)
{
	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;

	int stack_pointer													= 0;
	unsigned int node_index_stack[LightTreeATSSplittingMaxLightSamples] = { 0 };
	unsigned int light_samples_counter									= 1;

	int node_indices_to_be_traversed_sp									   = 0;
	int node_indices_to_be_traversed[LightTreeATSSplittingMaxLightSamples] = { -1 };

	while (light_samples_counter < LightTreeATSSplittingMaxLightSamples && stack_pointer >= 0)
	{
		unsigned int node_index				= node_index_stack[stack_pointer--];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		light_samples_counter--;

		// TODO is this node importance check needed
		float node_importance = light_tree_ats_node_importance<UseOrientation>(current_node, shading_point, shading_normal);
		if (node_importance > 0.0f)
		{
			float node_coherence = light_tree_ats_node_coherence(current_node, shading_point);
			if (node_coherence < render_data.light_tree_ats.settings.light_tree_ats_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold exceeded, exploring both branches of the tree

				float node_importance_left =
					light_tree_ats_node_importance<UseOrientation>(nodes[current_node.left_child_index_or_first_triangle_index], shading_point, shading_normal);
				float node_importance_right = light_tree_ats_node_importance<UseOrientation>(nodes[current_node.left_child_index_or_first_triangle_index + 1],
																							 shading_point, shading_normal);

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
					if (stack_pointer < LightTreeATSSplittingMaxLightSamples - 1 && node_importance_left > 0.0f)
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
					if (stack_pointer < LightTreeATSSplittingMaxLightSamples - 1 && node_importance_right > 0.0f)
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

	LightSampleArray<DirectLightSampleCount<LSS_BASE_LIGHT_TREE_ATS>()> light_samples_out;

	while (stack_pointer >= 0)
	{
		unsigned int node_index				= node_index_stack[stack_pointer--];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeATSNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

			float left_importance  = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
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
			int index					= current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index			= render_data.light_tree_ats.indices_array[index];
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
		unsigned int node_index				= node_indices_to_be_traversed[to_be_traversed_index++];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		float cumulative_probability = 1.0f;
		while (current_node.triangle_count == 0)
		{
			LightTreeATSNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
			LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

			float left_importance  = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
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

			int index					= current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
			int triangle_index			= render_data.light_tree_ats.indices_array[index];
			int emissive_triangle_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

			light_samples_out[light_samples_counter - 1].emissive_triangle_global_index = emissive_triangle_index;
			light_samples_out[light_samples_counter - 1].pdf							= cumulative_probability / current_node.triangle_count;
			light_samples_counter--;
		}
	}

	return light_samples_out;
}

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE void replay_splitting(const HIPRTRenderData& render_data,
								   const LightTreeATSNodeDevice* nodes,
								   unsigned int node_index,
								   unsigned int& collected_split_samples,
								   float3_t shading_point,
								   float3_t shading_normal)
{
	int stack_pointer													= 0;
	unsigned int node_index_stack[LightTreeATSSplittingMaxLightSamples] = { node_index };

	while (collected_split_samples < LightTreeATSSplittingMaxLightSamples && stack_pointer >= 0)
	{
		unsigned int node_index				= node_index_stack[stack_pointer--];
		LightTreeATSNodeDevice current_node = nodes[node_index];

		collected_split_samples--;

		// TODO is this node importance check needed
		float node_importance = light_tree_ats_node_importance<UseOrientation>(current_node, shading_point, shading_normal);
		if (node_importance > 0.0f)
		{
			float node_coherence = light_tree_ats_node_coherence(current_node, shading_point);
			if (node_coherence < render_data.light_tree_ats.settings.light_tree_ats_splitting_variance && current_node.triangle_count == 0)
			{
				// Variance threshold exceeded, exploring both branches of the tree

				float node_importance_left =
					light_tree_ats_node_importance<UseOrientation>(nodes[current_node.left_child_index_or_first_triangle_index], shading_point, shading_normal);
				float node_importance_right = light_tree_ats_node_importance<UseOrientation>(nodes[current_node.left_child_index_or_first_triangle_index + 1],
																							 shading_point, shading_normal);

				if (node_importance_left > node_importance_right)
				{
					// We're going to want to explore the left child first

					// So inserting the right child first
					if (node_importance_right > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
						collected_split_samples++;
					}

					// And then the left child such that the left child is popped first
					// and explored first
					if (stack_pointer < LightTreeATSSplittingMaxLightSamples - 1 && node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						collected_split_samples++;
					}
				}
				else
				{
					// We're going to want to explore the right child first

					// So inserting the left child first
					if (node_importance_left > 0.0f)
					{
						node_index_stack[++stack_pointer] = current_node.left_child_index_or_first_triangle_index;
						collected_split_samples++;
					}

					// And then the right child such that the right child is popped first
					// and explored first
					if (stack_pointer < LightTreeATSSplittingMaxLightSamples - 1 && node_importance_right > 0.0f)
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

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data,
														   float3_t shading_point,
														   float3_t shading_normal,
														   int global_emissive_triangle_index)
{
	if (global_emissive_triangle_index == -1)
		return 0.0f;

	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;
	LightTreeATSNodeDevice current_node = nodes[0];

	float root_node_importance = light_tree_ats_node_importance<UseOrientation>(current_node, shading_point, shading_normal);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail												  = render_data.light_tree_ats.bit_trails[global_emissive_triangle_index];
	unsigned char current_depth											  = 0;
	unsigned int collected_split_samples								  = 1;
	int split_replay_stack_pointer										  = -1;
	unsigned int split_replay_stack[LightTreeATSSplittingMaxLightSamples] = { 0 };
	// We're going to disable splitting after encountering
	// the first node that isn't split
	bool can_split = true;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		LightTreeATSNodeDevice left_child  = nodes[current_node.left_child_index_or_first_triangle_index];
		LightTreeATSNodeDevice right_child = nodes[current_node.left_child_index_or_first_triangle_index + 1];

		float left_importance  = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
		float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return 0.0f;

		bool node_split = false;
		if (collected_split_samples < LightTreeATSSplittingMaxLightSamples && can_split)
		{
			float node_coherence = light_tree_ats_node_coherence(current_node, shading_point);
			node_split			 = node_coherence < render_data.light_tree_ats.settings.light_tree_ats_splitting_variance && current_node.triangle_count == 0;
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

				if (split_replay_stack_pointer < LightTreeATSSplittingMaxLightSamples - 1)
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

				if (split_replay_stack_pointer < LightTreeATSSplittingMaxLightSamples - 1)
				{
					split_replay_stack[++split_replay_stack_pointer] = current_node.left_child_index_or_first_triangle_index + 1;
					collected_split_samples++;
				}
			}

			bool exploring_left_child_first = left_importance > right_importance;
			bool needs_to_go_right_child	= (bit_trail & (1 << current_depth)) != 0;

			// In any of the two cases below, the splitting code that samples would have
			// explored the left or right child first but the bittrail replay wants to go
			// down the other node.
			//
			// Because of that, it is possible that splitting during sampling produced the
			// maximum number of samples from that child explored first, which means that we wouldn't
			// get any more splitting from the other child since the maximum number of samples was reached.
			// And this influences the computation of the PDF. So we need to replay the splitting in the
			// child that splitting would have chosen first, to be sure that we get the same number of splits
			// and that the PDF computation is correct.

			if ((exploring_left_child_first && needs_to_go_right_child) || (!exploring_left_child_first && !needs_to_go_right_child))
				replay_splitting<UseOrientation>(render_data, nodes, split_replay_stack[split_replay_stack_pointer--], collected_split_samples, shading_point,
												 shading_normal);

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

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE LightSampleArray<1> sample_one_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data,
																			 float3_t shading_point,
																			 float3_t view_direction,
																			 float3_t shading_normal,
																			 float3_t geometric_normal,
																			 int last_hit_primitive_index,
																			 RayPayload& ray_payload,
																			 Xorshift32Generator& rng)
{
	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;

	unsigned int current_node_index = 0;
	unsigned int depth				= 0;

	float cumulative_probability = 1.0f;
	while (nodes[current_node_index].triangle_count == 0)
	{
		const LightTreeATSNodeDevice& current_node = nodes[current_node_index];
		const unsigned int left_index			   = current_node.left_child_index_or_first_triangle_index;
		const unsigned int right_index			   = left_index + 1;
		const LightTreeATSNodeDevice& left_child   = nodes[left_index];
		const LightTreeATSNodeDevice& right_child  = nodes[right_index];

		float left_importance  = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
		float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return LightSampleArray<1>{ LightSampleInformation() };

		float p_left				  = left_importance / (left_importance + right_importance);
		const float cumulative_before = cumulative_probability;

		if (rng() < p_left)
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

	const LightTreeATSNodeDevice& current_node = nodes[current_node_index];
	int index								   = current_node.left_child_index_or_first_triangle_index + rng.random_index(current_node.triangle_count);
	int triangle_index						   = render_data.light_tree_ats.indices_array[index];
	int emissive_triangle_index				   = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = emissive_triangle_index;
	light_sample.pdf							= cumulative_probability * (1.0f / current_node.triangle_count); // Sampling that triangle in that node

	return LightSampleArray<1>{ light_sample };
}

template <bool UseOrientation = LightTreeATSImportanceFunctionUseOrientation>
HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_ats(const HIPRTRenderData& render_data,
														   float3_t shading_point,
														   float3_t shading_normal,
														   int global_emissive_triangle_index)
{
	if (global_emissive_triangle_index == -1)
		return 0.0f;

	const LightTreeATSNodeDevice* nodes = render_data.light_tree_ats.nodes;

	LightTreeATSNodeDevice current_node = nodes[0];
	unsigned int current_node_index		= 0;

	float root_node_importance = light_tree_ats_node_importance<UseOrientation>(current_node, shading_point, shading_normal);
	if (root_node_importance <= 0.0f)
		return 0.0f;

	unsigned int bit_trail	   = render_data.light_tree_ats.bit_trails[global_emissive_triangle_index];
	unsigned int current_depth = 0;

	float cumulative_probability = 1.0f;
	while (current_node.triangle_count == 0)
	{
		const unsigned int left_index	   = current_node.left_child_index_or_first_triangle_index;
		const unsigned int right_index	   = left_index + 1;
		LightTreeATSNodeDevice left_child  = nodes[left_index];
		LightTreeATSNodeDevice right_child = nodes[right_index];

		float left_importance  = light_tree_ats_node_importance<UseOrientation>(left_child, shading_point, shading_normal);
		float right_importance = light_tree_ats_node_importance<UseOrientation>(right_child, shading_point, shading_normal);
		if (left_importance == 0.0f && right_importance == 0.0f)
			return 0.0f;

		float p_left					= left_importance / (left_importance + right_importance);
		bool target_is_left				= !(bit_trail & (1 << current_depth));
		float target_branch_probability = target_is_left ? p_left : 1.0f - p_left;

		cumulative_probability *= target_branch_probability;

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
	return cumulative_probability / current_node.triangle_count;
}

#endif

#endif
