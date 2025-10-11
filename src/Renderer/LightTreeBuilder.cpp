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

float3 LightTreeBuilder::get_triangle_vertex(unsigned int linear_emissive_triangle_index, unsigned int vertex_index, const BuilderTrianglesData& triangles_data) const
{
	int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
	return triangles_data.vertices_positions[triangles_data.triangle_vertex_indices[triangles_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + vertex_index]];
}

void LightTreeBuilder::build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_vertex_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
{
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Building light tree...");
	auto start = std::chrono::high_resolution_clock::now();

	BuilderTrianglesData triangles_data(emissive_triangles_primitive_indices, triangle_vertex_indices, vertices_positions, material_indices, materials);

	m_nodes.resize(emissive_triangles_primitive_indices.size() * 2 - 1);
	m_triangle_indices.resize(emissive_triangles_primitive_indices.size());
	m_bit_trails.resize(emissive_triangles_primitive_indices.size(), 0u);
	m_bins_temp_buffer.resize(m_build_options.bin_count);
	m_left_bins_info_temp_buffer.resize(m_build_options.bin_count);
	m_right_bins_info_temp_buffer.resize(m_build_options.bin_count);
	std::iota(m_triangle_indices.begin(), m_triangle_indices.end(), 0);

	m_prefetched_triangles.resize(emissive_triangles_primitive_indices.size());

	for (int i = 0; i < emissive_triangles_primitive_indices.size(); i++)
	{
		float3 v0 = get_triangle_vertex(i, 0, triangles_data);
		float3 v1 = get_triangle_vertex(i, 1, triangles_data);
		float3 v2 = get_triangle_vertex(i, 2, triangles_data);

		float3 triangle_normal = hippt::cross(v1 - v0, v2 - v0);
		float normal_length = hippt::length(triangle_normal);
		float triangle_area = 0.0f;
		if (normal_length > TriangleSamplingNormalLengthRejectionThreshold)
			triangle_area = normal_length * 0.5f;
		triangle_normal /= normal_length;

		const CPUMaterial& mat = triangles_data.materials[triangles_data.material_indices[triangles_data.emissive_triangles_primitive_indices[i]]];

		m_prefetched_triangles[i].bounds.extend(v0);
		m_prefetched_triangles[i].bounds.extend(v1);
		m_prefetched_triangles[i].bounds.extend(v2);

		m_prefetched_triangles[i].centroid = (v0 + v1 + v2) / 3.0f;
		m_prefetched_triangles[i].normal = triangle_normal;
		m_prefetched_triangles[i].area = triangle_area;
		m_prefetched_triangles[i].power = mat.get_total_emission() * triangle_area;
	}

	m_current_node_index = 0;

	LightTreeNode& root = m_nodes[m_current_node_index];
	root.left_child_index = 0;
	root.first_triangle_index = 0;
	root.triangle_count = (unsigned int)emissive_triangles_primitive_indices.size();

	update_node_bounds(m_current_node_index, triangles_data);
	subdivide_node(m_current_node_index++, triangles_data, 0);

	auto stop = std::chrono::high_resolution_clock::now();
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Light tree construction time: %ldms", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void LightTreeBuilder::update_node_bounds(unsigned int node_index, const BuilderTrianglesData& triangles_data)
{
	LightTreeNode& node = m_nodes[node_index];
	node.node_bounds.mini = float3(1e30f, 1e30f, 1e30f);
	node.node_bounds.maxi = float3(-1e30f, -1e30f, -1e30f);

	for (unsigned int first = node.first_triangle_index, i = 0; i < node.triangle_count; i++)
	{
		int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(first + i);

		float triangle_area = m_prefetched_triangles[emissive_triangle_index].area;
		if (triangle_area == 0.0f)
			continue;
		float3 triangle_normal = m_prefetched_triangles[emissive_triangle_index].normal;

		node.node_bounds.extend(m_prefetched_triangles[emissive_triangle_index].bounds);
		node.total_power += m_prefetched_triangles[emissive_triangle_index].power;
		node.cone_union_with(triangle_normal, 0.0f, (float)M_PI / 2.0f);
	}
}

void LightTreeBuilder::subdivide_node(unsigned int node_index, const BuilderTrianglesData& triangles_data, int depth)
{
	LightTreeNode& node = m_nodes[node_index];
	if (node.triangle_count <= m_build_options.max_triangles_per_leaf)
	{
		register_node_bit_trail(node, triangles_data);

		return;
	}

	int split_axis;
	float split_position;
	float split_cost = compute_split_position(node, split_axis, split_position, triangles_data);

	if (m_build_options.cost_function == LIGHT_TREE_BUILD_COST_FUNCTION_SAH)
	{
		float no_split_cost = compute_node_cost(node);
		if (split_cost >= no_split_cost)
		{
			register_node_bit_trail(node, triangles_data);

			return;
		}
	}
	else if (m_build_options.cost_function == LIGHT_TREE_BUILD_COST_FUNCTION_SAOH)
	{
		if (node.triangle_count <= 1)
		{
			register_node_bit_trail(node, triangles_data);

			return;
		}
	}

	int right_node_start = partition_node_primitives(node_index, split_axis, split_position);
	int left_count = right_node_start - node.first_triangle_index;
	if (left_count == 0 || // Zero triangles on the left
		left_count == node.triangle_count) // Zero triangles on the right
	{
		register_node_bit_trail(node, triangles_data);

		return;
	}

	int left_child_index = m_current_node_index++;
	int right_child_index = m_current_node_index++;

	LightTreeNode& left_child = m_nodes[left_child_index];
	left_child.first_triangle_index = node.first_triangle_index;
	left_child.triangle_count = left_count;
	left_child.bit_trail = node.bit_trail;

	LightTreeNode& right_child = m_nodes[right_child_index];
	right_child.first_triangle_index = right_node_start;
	right_child.triangle_count = node.triangle_count - left_count;
	right_child.bit_trail = node.bit_trail;
	right_child.bit_trail |= 1 << depth;

	node.left_child_index = left_child_index;
	node.triangle_count = 0;

	update_node_bounds(left_child_index, triangles_data);
	update_node_bounds(right_child_index, triangles_data);

	subdivide_node(left_child_index, triangles_data, depth + 1);
	subdivide_node(right_child_index, triangles_data, depth + 1);
}

float LightTreeBuilder::compute_saoh_m_omega(const LightTreeNodeOrientationData& orientation_data) const
{
	float sin_theta_o = sinf(orientation_data.theta_o);
	float cos_theta_o = cosf(orientation_data.theta_o);// sqrtf(1.0f - hippt::square(sin_theta_o));
	float theta_w = hippt::min(orientation_data.theta_o + orientation_data.theta_e, static_cast<float>(M_PI));
	return 2.0f * M_PI * (1.0f - cos_theta_o) + M_PI * 0.5f * (2.0f * theta_w * sin_theta_o - cosf(orientation_data.theta_o - 2.0f * theta_w) - 2.0f * orientation_data.theta_o * sin_theta_o + cos_theta_o);
}

float LightTreeBuilder::compute_split_position(const LightTreeNode& node, int& out_split_axis, float& out_split_position, const BuilderTrianglesData& triangles_data)
{
	if (m_build_options.build_split_method == LIGHT_TREE_BUILD_OPTION_SPLIT_MIDPOINT)
	{
		if (node.triangle_count <= 2)
			return 1.0e30f;

		float3 extents = node.node_bounds.get_extents();

		int split_axis = extents.y > extents.x ? (extents.z > extents.y ? 2 : 1) : (extents.z > extents.x ? 2 : 0);
		float split_position = hippt::idx(extents, split_axis) * 0.5f + hippt::idx(node.node_bounds.mini, split_axis);

		out_split_axis = split_axis;
		out_split_position = split_position;

		return 0.0f;
	}
	else if (m_build_options.build_split_method == LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED)
	{
		int best_axis = -1;
		float best_position = 0.0f;
		float best_cost = 1.0e30f;

		AABB prims_bounds;

		for (unsigned int first = node.first_triangle_index, i = 0; i < node.triangle_count; i++)
		{
			int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(first + i);

			prims_bounds.extend(m_prefetched_triangles[emissive_triangle_index].bounds);
		}

		// For SAOH
		float3 bounds_extents = prims_bounds.get_extents();
		float longest_bounds_axis = hippt::max(bounds_extents.x, hippt::max(bounds_extents.y, bounds_extents.z));
		float node_m_omega = compute_saoh_m_omega(node.orientation_data);

		for (int axis_index = 0; axis_index < 3; axis_index++)
		{
			if (hippt::idx(prims_bounds.maxi, axis_index) == hippt::idx(prims_bounds.mini, axis_index))
				continue;

			std::fill(m_bins_temp_buffer.begin(), m_bins_temp_buffer.end(), Bin());
			std::fill(m_left_bins_info_temp_buffer.begin(), m_left_bins_info_temp_buffer.end(), BinCostInfo());
			std::fill(m_right_bins_info_temp_buffer.begin(), m_right_bins_info_temp_buffer.end(), BinCostInfo());

			// Computing the bounds of the bins
			// TODO bin 3 axis at the same time
			float scale = m_build_options.bin_count / hippt::idx(prims_bounds.maxi - prims_bounds.mini, axis_index);
			for (unsigned int first = node.first_triangle_index, i = 0; i < node.triangle_count; i++)
			{
				int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(first + i);

				float triangle_area = m_prefetched_triangles[emissive_triangle_index].area;
				if (triangle_area == 0.0f)
					continue;

				float3 triangle_normal = m_prefetched_triangles[emissive_triangle_index].normal;

				float3 centroid = m_prefetched_triangles[emissive_triangle_index].centroid;
				int bin_index_no_clamp = (int)((hippt::idx(centroid, axis_index) - hippt::idx(prims_bounds.mini, axis_index)) * scale);
				int bin_index = hippt::clamp(0, m_build_options.bin_count - 1, bin_index_no_clamp);

				m_bins_temp_buffer[bin_index].bounds.extend(m_prefetched_triangles[emissive_triangle_index].bounds);
				m_bins_temp_buffer[bin_index].tri_count++;

				m_bins_temp_buffer[bin_index].total_power += m_prefetched_triangles[emissive_triangle_index].power;
				m_bins_temp_buffer[bin_index].orientation_data.cone_union_with(triangle_normal, 0.0f, static_cast<float>(M_PI) / 2.0f);
			}

			AABB left_box, right_box;
			LightTreeNodeOrientationData left_orientation_data, right_orientation_data;
			ColorRGB32F left_energy, right_energy;
			int left_tri_count_sum = 0, right_tri_count_sum = 0;
			for (int i = 0; i < m_build_options.bin_count - 1; i++)
			{
				left_tri_count_sum += m_bins_temp_buffer[i].tri_count;
				left_box.extend(m_bins_temp_buffer[i].bounds);
				left_energy += m_bins_temp_buffer[i].total_power;
				left_orientation_data.cone_union_with(m_bins_temp_buffer[i].orientation_data);

				m_left_bins_info_temp_buffer[i].tri_count = left_tri_count_sum;
				m_left_bins_info_temp_buffer[i].area = left_box.area();
				m_left_bins_info_temp_buffer[i].energy = left_energy.luminance();
				m_left_bins_info_temp_buffer[i].m_omega = compute_saoh_m_omega(left_orientation_data);



				right_tri_count_sum += m_bins_temp_buffer[m_build_options.bin_count - 1 - i].tri_count;
				right_box.extend(m_bins_temp_buffer[m_build_options.bin_count - 1 - i].bounds);
				right_energy += m_bins_temp_buffer[m_build_options.bin_count - 1 - i].total_power;
				right_orientation_data.cone_union_with(m_bins_temp_buffer[m_build_options.bin_count - 1 - i].orientation_data);

				m_right_bins_info_temp_buffer[m_build_options.bin_count - 2 - i].tri_count = right_tri_count_sum;
				m_right_bins_info_temp_buffer[m_build_options.bin_count - 2 - i].area = right_box.area();
				m_right_bins_info_temp_buffer[m_build_options.bin_count - 2 - i].energy = right_energy.luminance();
				m_right_bins_info_temp_buffer[m_build_options.bin_count - 2 - i].m_omega = compute_saoh_m_omega(right_orientation_data);
			}


			float scale_sah = hippt::idx(prims_bounds.maxi - prims_bounds.mini, axis_index) / m_build_options.bin_count;
			for (int split_plane_index = 0; split_plane_index < m_build_options.bin_count; split_plane_index++)
			{
				if (m_left_bins_info_temp_buffer[split_plane_index].tri_count == 0 || m_right_bins_info_temp_buffer[split_plane_index].tri_count == 0)
					continue;

				float candidate_split_position = hippt::idx(prims_bounds.mini, axis_index) + (split_plane_index + 1) * scale_sah;

				float cost = 1.0e30f;
				if (m_build_options.cost_function == LIGHT_TREE_BUILD_COST_FUNCTION_SAOH)
				{
					cost = 0.0f;
					
					// Numerator
					cost += m_left_bins_info_temp_buffer[split_plane_index].area * m_left_bins_info_temp_buffer[split_plane_index].energy * m_left_bins_info_temp_buffer[split_plane_index].m_omega;
					cost += m_right_bins_info_temp_buffer[split_plane_index].area * m_right_bins_info_temp_buffer[split_plane_index].energy * m_right_bins_info_temp_buffer[split_plane_index].m_omega;

					// Denominator
					cost /= prims_bounds.area() * node_m_omega;

					// Kr
					float Kr = longest_bounds_axis / hippt::idx(bounds_extents, axis_index);
					cost *= Kr;
				}
				else if (m_build_options.cost_function == LIGHT_TREE_BUILD_COST_FUNCTION_SAH)
				{
					cost = m_left_bins_info_temp_buffer[split_plane_index].tri_count * m_left_bins_info_temp_buffer[split_plane_index].area +
						   m_right_bins_info_temp_buffer[split_plane_index].tri_count * m_right_bins_info_temp_buffer[split_plane_index].area;
				}

				if (cost < best_cost)
				{
					best_axis = axis_index;
					best_position = candidate_split_position;
					best_cost = cost;
				}
			}
		}

		out_split_axis = best_axis;
		out_split_position = best_position;

		return best_cost;
	}
	else
		return 0.0f;
}

float LightTreeBuilder::compute_node_cost(const LightTreeNode& node)
{
	float area = node.node_bounds.area();
	float cost = node.triangle_count * area;

	return cost;
}

float LightTreeBuilder::compute_sah_cost(const LightTreeNode& node, int axis_index, float split_position, const BuilderTrianglesData& triangles_data)
{
	AABB box_left;
	AABB box_right;

	unsigned int triangle_count_left = 0;
	unsigned int triangle_count_right = 0;

	unsigned int first_triangle_index = node.first_triangle_index;
	for (int triangle_index = 0; triangle_index < node.triangle_count; triangle_index++)
	{
		int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(first_triangle_index + triangle_index);
		if (hippt::idx(m_prefetched_triangles[emissive_triangle_index].centroid, axis_index) < split_position)
		{
			box_left.extend(m_prefetched_triangles[emissive_triangle_index].bounds);

			triangle_count_left++;
		}
		else
		{
			box_right.extend(m_prefetched_triangles[emissive_triangle_index].bounds);

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

		if (hippt::idx(m_prefetched_triangles[centroid_index].centroid, axis) < split_position)
			start++;
		else
			std::swap(m_triangle_indices[start], m_triangle_indices[end--]);
	}

	return start;
}

void LightTreeBuilder::register_node_bit_trail(const LightTreeNode& node, const BuilderTrianglesData& data)
{
	for (int triangle_index = 0; triangle_index < node.triangle_count; triangle_index++)
		m_bit_trails[node.first_triangle_index + triangle_index] = node.bit_trail;
}

void LightTreeBuilder::cleanup()
{
	m_nodes = std::vector<LightTreeNode>();
	m_prefetched_triangles = std::vector<PrefetchedTriangle>();
	m_triangle_indices = std::vector<int>();
}

LightTreeBuilderOptions& LightTreeBuilder::get_options()
{
	return m_build_options;
}
