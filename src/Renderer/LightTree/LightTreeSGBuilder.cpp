/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "LightTreeSGBuilder.h"

void LightTreeSGBuilder::build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
{
	m_light_tree_ats_builder.build_light_tree(emissive_triangles_primitive_indices, triangle_indices, vertices_positions, material_indices, materials);

	m_nodes.resize(m_light_tree_ats_builder.get_nodes().size());

	compute_node_spherical_gaussian(0, LightTreeBuilderTrianglesData(emissive_triangles_primitive_indices, triangle_indices, vertices_positions, material_indices, materials));
}

void LightTreeSGBuilder::compute_node_spherical_gaussian(unsigned int node_index, const LightTreeBuilderTrianglesData& triangle_data)
{
	const std::vector<LightTreeATSBuilder::PrefetchedTriangle>& prefetched_triangles = m_light_tree_ats_builder.get_prefetched_triangles();
	const std::vector<LightTreeATSNode>& ats_nodes = m_light_tree_ats_builder.get_nodes();
	const LightTreeATSNode& ats_node = ats_nodes[node_index];

	LightTreeSGNode& sg_node = m_nodes[node_index];

	if (ats_node.triangle_count == 0)
	{
		// Recurse children if not a leaf

		compute_node_spherical_gaussian(ats_node.left_child_index, triangle_data);
		compute_node_spherical_gaussian(ats_node.left_child_index + 1, triangle_data);

		LightTreeSGNode& left_node = m_nodes[ats_node.left_child_index];
		LightTreeSGNode& right_node = m_nodes[ats_node.left_child_index + 1];
		float left_weight = 1.0f, right_weight = 1.0f;
		if (left_node.total_power == 0.0f)
			left_weight = 0.0f;
		else if (right_node.total_power == 0.0f)
			right_weight = 0.0f;

		if (left_weight == 0.0f && right_weight == 0.0f)
			// Both children are empty
			return;

		left_weight = left_node.total_power / (left_node.total_power + right_node.total_power);
		right_weight = right_node.total_power / (left_node.total_power + right_node.total_power);

		sg_node.mean_axis = left_weight * left_node.mean_axis + right_weight * right_node.mean_axis;
		sg_node.total_power = left_node.total_power + right_node.total_power;
		sg_node.spatial_mean = left_weight * left_node.spatial_mean + right_weight * right_node.spatial_mean;
		sg_node.spatial_variance = left_weight * left_node.spatial_variance + right_weight * right_node.spatial_variance + left_weight * right_weight * hippt::length2(left_node.spatial_mean - right_node.spatial_mean);

		sg_node.compute_vmf();

		sg_node.left_child_index = ats_node.left_child_index;
		sg_node.triangle_count = 0;
	}
	else
	{
		// Leaf node, compute data necessary for building the spherical gaussian
		sg_node.mean_axis = make_float3(0.0f, 0.0f, 0.0f);
		sg_node.spatial_mean = make_float3(0.0f, 0.0f, 0.0f);
		sg_node.spatial_variance = 0.0f;
		sg_node.total_power = 0.0f;

		float4 sum_positions = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		unsigned int valid_triangle_count = 0;
		for (int i = 0; i < ats_node.triangle_count; i++)
		{
			unsigned int linear_emissive_triangle_index = ats_node.first_triangle_index + i;
			int emissive_triangle_index = m_light_tree_ats_builder.bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
			const LightTreeATSBuilder::PrefetchedTriangle& triangle = prefetched_triangles[emissive_triangle_index];
			if (triangle.area == 0.0f)
				// Degenerate triangle
				continue;

			// 0.5f * triangle normal from the paper
			sg_node.mean_axis += 0.5f * triangle.normal;
			sg_node.total_power += triangle.power;
			sg_node.spatial_mean += triangle.centroid;

			sum_positions += make_float4(triangle.centroid.x, triangle.centroid.y, triangle.centroid.z, hippt::dot(triangle.centroid, triangle.centroid)) * triangle.power;

			valid_triangle_count++;
		}

		if (valid_triangle_count == 0)
			return;

		sg_node.mean_axis /= valid_triangle_count;
		sg_node.spatial_mean /= valid_triangle_count;
		
		// Computing the variance of the leaf for all triangles it contains
		float4 positions_avg = sum_positions / sg_node.total_power;
		for (int i = 0; i < ats_node.triangle_count; i++)
		{
			unsigned int linear_emissive_triangle_index = ats_node.first_triangle_index + i;
			int emissive_triangle_index = m_light_tree_ats_builder.bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
			const LightTreeATSBuilder::PrefetchedTriangle& triangle = prefetched_triangles[emissive_triangle_index];

			if (triangle.area == 0.0f)
				continue;

			float3 p0 = triangle_data.vertices_positions[triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 0]];
			float3 p1 = triangle_data.vertices_positions[triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 1]];
			float3 p2 = triangle_data.vertices_positions[triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 2]];

			float3 edge1 = p1 - p0;
			float3 edge2 = p2 - p0;

			sg_node.spatial_variance = (positions_avg.w - hippt::dot(make_float3(positions_avg.x, positions_avg.y, positions_avg.z), make_float3(positions_avg.x, positions_avg.y, positions_avg.z)));
		}

		// Computes vMF parameters with the mean axis (normalized by the function)
		sg_node.compute_vmf();
		sg_node.triangle_count = valid_triangle_count;
		sg_node.first_triangle_index = ats_node.first_triangle_index;
	}
}

void LightTreeSGBuilder::cleanup()
{
	m_light_tree_ats_builder.cleanup();
	m_nodes.clear();
}
