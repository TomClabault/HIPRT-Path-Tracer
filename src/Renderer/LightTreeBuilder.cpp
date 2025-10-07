/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/LightTreeBuilder.h"

#include <numeric>

int LightTreeBuilder::bvh_triangle_index_to_emissive_triangle_index(int bvh_triangle_index) const
{
	return m_triangle_indices[bvh_triangle_index];
}

float3 LightTreeBuilder::get_triangle_vertex(unsigned int linear_emissive_triangle_index, unsigned int vertex_index, const BuilderTrianglesPayload& payload) const
{
	int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
	return payload.vertices_positions[payload.triangle_vertex_indices[payload.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + vertex_index]];
}

void LightTreeBuilder::build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_vertex_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
{
	auto start = std::chrono::high_resolution_clock::now();

	BuilderTrianglesPayload triangles_payload(emissive_triangles_primitive_indices, triangle_vertex_indices, vertices_positions, material_indices, materials);

	m_nodes.resize(emissive_triangles_primitive_indices.size() * 2 - 1);
	m_centroids.resize(emissive_triangles_primitive_indices.size());
	m_triangle_indices.resize(emissive_triangles_primitive_indices.size());
	std::iota(m_triangle_indices.begin(), m_triangle_indices.end(), 0);


	//#pragma omp parallel for
	for (int i = 0; i < emissive_triangles_primitive_indices.size(); i++)
	{
		float3 v0 = get_triangle_vertex(i, 0, triangles_payload);
		float3 v1 = get_triangle_vertex(i, 1, triangles_payload);
		float3 v2 = get_triangle_vertex(i, 2, triangles_payload);

		m_centroids[i] = (v0 + v1 + v2) * 0.3333333f;
	}

	LightTreeNode& root = m_nodes[m_current_node_index];
	root.left_child_index = 0;
	root.first_triangle_index = 0;
	root.triangle_count = (unsigned int)emissive_triangles_primitive_indices.size();

	update_node_bounds(m_current_node_index, triangles_payload);
	subdivide_node(m_current_node_index++, triangles_payload);

	auto stop = std::chrono::high_resolution_clock::now();
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Light tree construction time: %ldms", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void LightTreeBuilder::update_node_bounds(unsigned int node_index, const BuilderTrianglesPayload& triangles_payload)
{
	LightTreeNode& node = m_nodes[node_index];
	node.node_bounds.mini = float3(1e30f, 1e30f, 1e30f);
	node.node_bounds.maxi = float3(-1e30f, -1e30f, -1e30f);

	for (unsigned int first = node.first_triangle_index, i = 0; i < node.triangle_count; i++)
	{
		float3 v0 = get_triangle_vertex(first + i, 0, triangles_payload);
		float3 v1 = get_triangle_vertex(first + i, 1, triangles_payload);
		float3 v2 = get_triangle_vertex(first + i, 2, triangles_payload);

		node.node_bounds.extend(v0);
		node.node_bounds.extend(v1);
		node.node_bounds.extend(v2);

		int index = bvh_triangle_index_to_emissive_triangle_index(first + i);
		const CPUMaterial& mat = triangles_payload.materials[triangles_payload.material_indices[triangles_payload.emissive_triangles_primitive_indices[index]]];

		float3 triangle_normal = hippt::cross(v1 - v0, v2 - v0);
		float triangle_area = hippt::length(triangle_normal) * 0.5f;
		triangle_normal /= triangle_area * 2.0f;

		node.total_power += mat.emission * mat.emission_strength * triangle_area;
		node.cones_union(triangle_normal, 0.0f, (float)M_PI / 2.0f);
	}
}

void LightTreeBuilder::subdivide_node(unsigned int node_index, const BuilderTrianglesPayload& triangles_payload)
{
	LightTreeNode& node = m_nodes[node_index];

	int split_axis;
	float split_position;
	float split_cost = compute_split_position(node, split_axis, split_position, triangles_payload);

	float no_split_cost = compute_node_cost(node);
	if (split_cost >= no_split_cost)
		return;

	int right_node_start = partition_node_primitives(node_index, split_axis, split_position);
	int left_count = right_node_start - node.first_triangle_index;
	if (left_count == 0 || // Zero triangles on the left
		left_count == node.triangle_count) // Zero triangles on the right
		return;

	int left_child_index = m_current_node_index++;
	int right_child_index = m_current_node_index++;

	LightTreeNode& left_child = m_nodes[left_child_index];
	left_child.first_triangle_index = node.first_triangle_index;
	left_child.triangle_count = left_count;

	LightTreeNode& right_child = m_nodes[right_child_index];
	right_child.first_triangle_index = right_node_start;
	right_child.triangle_count = node.triangle_count - left_count;

	node.left_child_index = left_child_index;
	node.triangle_count = 0;

	update_node_bounds(left_child_index, triangles_payload);
	update_node_bounds(right_child_index, triangles_payload);

	subdivide_node(left_child_index, triangles_payload);
	subdivide_node(right_child_index, triangles_payload);
}

#define SPLIT_MIDPOINT 0
#define SPLIT_BINNED_SAH 1
#define SPLIT_SAOH 2

#define SPLIT_METHOD SPLIT_BINNED_SAH

#define BINNED_SAH_BIN_COUNT 100

float LightTreeBuilder::compute_split_position(const LightTreeNode& node, int& out_split_axis, float& out_split_position, const BuilderTrianglesPayload& triangles_payload)
{
#if SPLIT_METHOD == SPLIT_MIDPOINT
	float3 extents = node.node_bounds.get_extents();

	int split_axis = extents.y > extents.x ? (extents.z > extents.y ? 2 : 1) : (extents.z > extents.x ? 2 : 0);
	float split_position = hippt::idx(extents, split_axis) * 0.5f + hippt::idx(node.node_bounds.mini, split_axis);

	return 0.0f;
#elif SPLIT_METHOD == SPLIT_BINNED_SAH
	int best_axis = -1;
	float best_position = 0.0f;
	float best_cost = 1.0e30f;

	for (int axis_index = 0; axis_index < 3; axis_index++)
	{
		float scale = hippt::idx(node.node_bounds.maxi - node.node_bounds.mini, axis_index) / BINNED_SAH_BIN_COUNT;
		for (int split_plane_index = 0; split_plane_index < BINNED_SAH_BIN_COUNT; split_plane_index++)
		{
			float candidate_split_position = hippt::idx(node.node_bounds.mini, axis_index) + split_plane_index * scale;
			float sah_cost = compute_sah_cost(node, axis_index, candidate_split_position, triangles_payload);
			if (sah_cost < best_cost)
			{
				best_axis = axis_index;
				best_position = candidate_split_position;
				best_cost = sah_cost;
			}
		}
	}

	out_split_axis = best_axis;
	out_split_position = best_position;

	return best_cost;
#endif
}

float LightTreeBuilder::compute_node_cost(const LightTreeNode& node)
{
	float area = node.node_bounds.area();
	float cost = node.triangle_count * area;

	return cost;
}

float LightTreeBuilder::compute_sah_cost(const LightTreeNode& node, int axis_index, float split_position, const BuilderTrianglesPayload& triangles_payload)
{
	AABB box_left;
	AABB box_right;

	unsigned int triangle_count_left = 0;
	unsigned int triangle_count_right = 0;

	unsigned int first_triangle_index = node.first_triangle_index;
	for (int triangle_index = 0; triangle_index < node.triangle_count; triangle_index++)
	{
		int bvh_triangle_index = first_triangle_index + triangle_index;
		if (hippt::idx(m_centroids[bvh_triangle_index_to_emissive_triangle_index(bvh_triangle_index)], axis_index) < split_position)
		{
			box_left.extend(get_triangle_vertex(bvh_triangle_index, 0, triangles_payload));
			box_left.extend(get_triangle_vertex(bvh_triangle_index, 1, triangles_payload));
			box_left.extend(get_triangle_vertex(bvh_triangle_index, 2, triangles_payload));

			triangle_count_left++;
		}
		else
		{
			box_right.extend(get_triangle_vertex(bvh_triangle_index, 0, triangles_payload));
			box_right.extend(get_triangle_vertex(bvh_triangle_index, 1, triangles_payload));
			box_right.extend(get_triangle_vertex(bvh_triangle_index, 2, triangles_payload));

			triangle_count_right++;
		}
	}

	float cost = triangle_count_left * box_left.area() + triangle_count_right * box_right.area();

	return cost > 0 ? cost : 1.0e30f;
}

int LightTreeBuilder::partition_node_primitives(unsigned int node_index, int axis, float split_position)
{
	LightTreeNode& node = m_nodes[node_index];

	int start = node.first_triangle_index;
	int end = start + node.triangle_count - 1;

	while (start <= end)
	{
		int centroid_index = bvh_triangle_index_to_emissive_triangle_index(start);

		if (hippt::idx(m_centroids[centroid_index], axis) < split_position)
			start++;
		else
			std::swap(m_triangle_indices[start], m_triangle_indices[end--]);
	}

	return start;
}

void LightTreeBuilder::cleanup()
{
	m_nodes.clear();
	m_centroids.clear();
	m_triangle_indices.clear();
}
