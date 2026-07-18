/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/LightTree/LightTreeSGBuilder.h"

void LightTreeSGBuilder::build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices,
										  const std::vector<float>& triangles_average_emissive_power_luminance,
										  const std::vector<int>& triangle_indices,
										  const std::vector<float3_t>& vertices_positions)
{
	m_light_tree_ats_builder.build_light_tree(emissive_triangles_primitive_indices, triangles_average_emissive_power_luminance, triangle_indices,
											  vertices_positions);

	m_nodes.resize(m_light_tree_ats_builder.get_nodes().size());
	if (m_nodes.empty())
		return;

	compute_node_spherical_gaussian(0, LightTreeBuilderTrianglesData(emissive_triangles_primitive_indices, triangle_indices, vertices_positions));
}

void LightTreeSGBuilder::compute_node_spherical_gaussian(unsigned int node_index, const LightTreeBuilderTrianglesData& triangle_data)
{
	const std::vector<LightTreeATSBuilder::PrefetchedTriangle>& prefetched_triangles = m_light_tree_ats_builder.get_prefetched_triangles();
	const std::vector<LightTreeATSNode>& ats_nodes									 = m_light_tree_ats_builder.get_nodes();
	const LightTreeATSNode& ats_node												 = ats_nodes[node_index];

	LightTreeSGNode& sg_node = m_nodes[node_index];

	if (ats_node.triangle_count == 0)
	{
		// Recurse children if not a leaf

		compute_node_spherical_gaussian(ats_node.left_child_index, triangle_data);
		compute_node_spherical_gaussian(ats_node.left_child_index + 1, triangle_data);

		LightTreeSGNode& left_node	= m_nodes[ats_node.left_child_index];
		LightTreeSGNode& right_node = m_nodes[ats_node.left_child_index + 1];

		float left_weight = 1.0f, right_weight = 1.0f;
		if (left_node.total_power == 0.0f)
			left_weight = 0.0f;
		if (right_node.total_power == 0.0f)
			right_weight = 0.0f;

		if (left_weight == 0.0f && right_weight == 0.0f)
		{
			// Both children are empty
			sg_node.total_power = 0.0f;

			return;
		}

		left_weight	 = left_node.total_power / (left_node.total_power + right_node.total_power);
		right_weight = right_node.total_power / (left_node.total_power + right_node.total_power);

		sg_node.orientation_axis = ats_node.orientation_data.axis;
		sg_node.theta_o			 = ats_node.orientation_data.theta_o;
		sg_node.mean_axis		 = left_weight * left_node.mean_axis + right_weight * right_node.mean_axis;
		sg_node.total_power		 = left_node.total_power + right_node.total_power;
		sg_node.bounds.extend(left_node.bounds);
		sg_node.bounds.extend(right_node.bounds);
		sg_node.spatial_mean			  = left_weight * left_node.spatial_mean + right_weight * right_node.spatial_mean;
		const float3_t mean_delta		  = left_node.spatial_mean - right_node.spatial_mean;
		const float3_t mean_delta_squared = make_float3(mean_delta.x * mean_delta.x, mean_delta.y * mean_delta.y, mean_delta.z * mean_delta.z);
		sg_node.spatial_variance_diag =
			left_weight * left_node.spatial_variance_diag + right_weight * right_node.spatial_variance_diag + left_weight * right_weight * mean_delta_squared;

		sg_node.total_emitter_count = left_node.total_emitter_count + right_node.total_emitter_count;
		if (sg_node.total_emitter_count > 0)
		{
			sg_node.energy_average = (left_node.total_emitter_count * left_node.energy_average + right_node.total_emitter_count * right_node.energy_average) /
									 sg_node.total_emitter_count;
			sg_node.energy_variance =
				hippt::max(0.0f, (left_node.total_emitter_count * (left_node.energy_variance + left_node.energy_average * left_node.energy_average) +
								  right_node.total_emitter_count * (right_node.energy_variance + right_node.energy_average * right_node.energy_average)) /
										 sg_node.total_emitter_count -
									 sg_node.energy_average * sg_node.energy_average);
		}

		sg_node.compute_vmf();

		float radius_squared = 0.0f;

		// Computing the bounding sphere of the node from the AABB of the node
		for (int corner_index = 0; corner_index < 8; ++corner_index)
		{
			const float3_t corner = make_float3((corner_index & 1) ? sg_node.bounds.mini.x : sg_node.bounds.maxi.x,
												(corner_index & 2) ? sg_node.bounds.mini.y : sg_node.bounds.maxi.y,
												(corner_index & 4) ? sg_node.bounds.mini.z : sg_node.bounds.maxi.z);

			radius_squared = hippt::max(radius_squared, hippt::length2(corner - sg_node.spatial_mean));
		}
		sg_node.bounding_sphere_radius = hippt::sqrt(radius_squared);

		sg_node.left_child_index = ats_node.left_child_index;
		sg_node.triangle_count	 = 0;
	}
	else
	{
		// Leaf node, compute data necessary for building the spherical gaussian
		sg_node.mean_axis	= make_float3(0.0f, 0.0f, 0.0f);
		sg_node.total_power = 0.0f;

		float3_t sum_positions			= make_float3(0.0f, 0.0f, 0.0f);
		float3_t sum_second_moment_diag = make_float3(0.0f, 0.0f, 0.0f);

		double sum_energy			= 0.0;
		double sum_energy_squared	= 0.0;
		unsigned int triangle_count = ats_node.triangle_count;
		for (int i = 0; i < ats_node.triangle_count; i++)
		{
			unsigned int linear_emissive_triangle_index = ats_node.first_triangle_index + i;
			int emissive_triangle_index = m_light_tree_ats_builder.bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
			const LightTreeATSBuilder::PrefetchedTriangle& triangle = prefetched_triangles[emissive_triangle_index];

			float3_t p0 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 0]];
			float3_t p1 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 1]];
			float3_t p2 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 2]];

			float3_t e1 = p1 - p0;
			float3_t e2 = p2 - p0;
			const float3_t triangle_variance_diag =
				make_float3((e1.x * e1.x + e2.x * e2.x - e1.x * e2.x) / 18.0f, (e1.y * e1.y + e2.y * e2.y - e1.y * e2.y) / 18.0f,
							(e1.z * e1.z + e2.z * e2.z - e1.z * e2.z) / 18.0f);
			const float3_t centroid_squared =
				make_float3(triangle.centroid.x * triangle.centroid.x, triangle.centroid.y * triangle.centroid.y, triangle.centroid.z * triangle.centroid.z);
			sum_second_moment_diag += (centroid_squared + triangle_variance_diag) * triangle.power;

			// 0.5f * triangle normal from the paper
			sg_node.mean_axis += 0.5f * triangle.normal * triangle.power;
			sg_node.total_power += triangle.power;

			sum_positions += triangle.centroid * triangle.power;

			sum_energy += triangle.power;
			sum_energy_squared += hippt::square(triangle.power);
		}

		sg_node.total_emitter_count = triangle_count;

		if (triangle_count == 0)
			return;
		else if (sg_node.total_power == 0.0f)
			return;

		sum_positions /= sg_node.total_power;
		sg_node.energy_average	= sum_energy / triangle_count;
		sg_node.energy_variance = hippt::max(0.0f, static_cast<float>(sum_energy_squared / triangle_count - hippt::square(sum_energy / triangle_count)));

		sg_node.mean_axis /= sg_node.total_power;
		sg_node.spatial_mean		  = sum_positions;
		const float3_t mean_squared	  = make_float3(sg_node.spatial_mean.x * sg_node.spatial_mean.x, sg_node.spatial_mean.y * sg_node.spatial_mean.y,
													sg_node.spatial_mean.z * sg_node.spatial_mean.z);
		sg_node.spatial_variance_diag = hippt::max(make_float3(0.0f, 0.0f, 0.0f), sum_second_moment_diag / sg_node.total_power - mean_squared);

		float bounding_sphere_radius = 0.0f;
		for (int i = 0; i < ats_node.triangle_count; i++)
		{
			unsigned int linear_emissive_triangle_index = ats_node.first_triangle_index + i;
			int emissive_triangle_index = m_light_tree_ats_builder.bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
			const LightTreeATSBuilder::PrefetchedTriangle& triangle = prefetched_triangles[emissive_triangle_index];

			float3_t p0 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 0]];
			float3_t p1 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 1]];
			float3_t p2 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 2]];

			sg_node.bounds.extend(p0);
			sg_node.bounds.extend(p1);
			sg_node.bounds.extend(p2);

			bounding_sphere_radius =
				hippt::max(bounding_sphere_radius, hippt::max(hippt::max(hippt::length(sg_node.spatial_mean - p0), hippt::length(sg_node.spatial_mean - p1)),
															  hippt::length(sg_node.spatial_mean - p2)));
		}

		// Computes vMF parameters with the mean axis (normalized by the function)
		sg_node.compute_vmf();
		sg_node.triangle_count		   = triangle_count;
		sg_node.first_triangle_index   = ats_node.first_triangle_index;
		sg_node.bounding_sphere_radius = bounding_sphere_radius;
		sg_node.orientation_axis	   = ats_node.orientation_data.axis;
		sg_node.theta_o				   = ats_node.orientation_data.theta_o;
	}
}

void LightTreeSGBuilder::cleanup()
{
	m_light_tree_ats_builder.cleanup();
	m_nodes.clear();
}

LightTreeATSBuilderOptions& LightTreeSGBuilder::get_build_options()
{
	return m_light_tree_ats_builder.get_build_options();
}
