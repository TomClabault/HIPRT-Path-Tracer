#ifndef RENDERER_LIGHT_TREE_BUILDER_H
#define RENDERER_LIGHT_TREE_BUILDER_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Material/MaterialCPU.h"
#include "HostDeviceCommon/RenderData.h"
#include "Scene/AABB.h"

#include <numeric>

template <template <typename> typename DataContainer>
class LightTreeBuilder
{
public:
	struct LightTreeNode
	{
		static constexpr float UNINITIALIZED_AXIS = -42.0f;

		static void cones_union(
			float3 axis_a, float theta_o_a, float theta_e_a,
			float3 axis_b, float theta_o_b, float theta_e_b,
			float3& axis_out, float& theta_o_out, float& theta_e_out)
		{
			if (axis_a.x == LightTreeNode::UNINITIALIZED_AXIS)
			{
				axis_out = axis_b;
				theta_o_out = theta_o_b;
				theta_e_out = theta_e_b;

				return;
			}

			if (theta_o_b > theta_o_a)
				std::swap(theta_o_a, theta_o_b);

			float theta_d = acos(hippt::dot(axis_a, axis_b));
			float theta_e = hippt::max(theta_e_a, theta_e_b);

			if (hippt::min(theta_d + theta_o_b, (float)M_PI) <= theta_o_a)
			{
				axis_out = axis_a;
				theta_o_out = theta_o_a;
				theta_e_out = theta_e;

				return;
			}
			else
			{
				float theta_o = (theta_o_a + theta_d + theta_o_b) / 2.0f;
				if (M_PI <= theta_o)
				{
					axis_out = axis_a;
					theta_o_out = M_PI;
					theta_e_out = theta_e;

					return;
				}

				float theta_r = theta_o - theta_o_a;
				float3 axis = rotate_vector(axis_a, hippt::cross(axis_a, axis_b), theta_r);


				axis_out = axis;
				theta_o_out = M_PI;
				theta_e_out = theta_e;

				return;
			}
		}

		void cones_union(float3 other_axis, float other_theta_o, float other_theta_e)
		{
			cones_union(
				axis, theta_o, theta_e, 
				other_axis, other_theta_o, other_theta_e,
				axis, theta_o, theta_e);
		}

		// Axis of the cluster
		float3 axis = make_float3(UNINITIALIZED_AXIS, UNINITIALIZED_AXIS, UNINITIALIZED_AXIS);
		// Normal bounds
		float theta_o;
		// Emission extents
		float theta_e;
		// Total emissive powxer of the node
		ColorRGB32F total_power;

		AABB node_bounds;
		unsigned int left_child_index;
		unsigned int first_triangle_index, triangle_count;
	};

	struct BuilderTrianglesPayload
	{
		BuilderTrianglesPayload(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_vertex_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
			: emissive_triangles_primitive_indices(emissive_triangles_primitive_indices)
			, triangle_vertex_indices(triangle_vertex_indices)
			, vertices_positions(vertices_positions)
			, material_indices(material_indices)
			, materials(materials) {}

		const std::vector<int>& emissive_triangles_primitive_indices;
		const std::vector<int>& triangle_vertex_indices;
		const std::vector<float3>& vertices_positions;
		const std::vector<int>& material_indices;
		const std::vector<CPUMaterial>& materials;
	};

	int bvh_triangle_index_to_emissive_triangle_index(int bvh_triangle_index) const;

	float3 get_triangle_vertex(unsigned int linear_emissive_triangle_index, unsigned int vertex_index, const BuilderTrianglesPayload& payload) const;

	void build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials);

	void update_node_bounds(unsigned int node_index, const BuilderTrianglesPayload& triangles_payload);
	void subdivide_node(unsigned int node_index, const BuilderTrianglesPayload& triangles_payload);
	bool compute_split_position(const LightTreeNode& node, int& out_split_axis, float& out_split_position, const BuilderTrianglesPayload& triangles_payload);
	float compute_sah_cost(const LightTreeNode& node, int axis_index, float split_position, const BuilderTrianglesPayload& triangles_payload);
	int partition_node_primitives(unsigned int node_index, int axis, float split_position);

	void to_device(HIPRTRenderData& render_data);

	/**
	 * Frees up the memory that was needed for building the tree
	 */
	void cleanup();

private:
	unsigned int m_current_node_index = 0;
	std::vector<LightTreeNode> m_nodes;

	std::vector<float3> m_centroids;
	std::vector<int> m_triangle_indices; // Indices of the emissive triangles from 0 to N - 1

	DataContainer<LightTreeNodeDevice> m_device_nodes_buffer;
	DataContainer<int> m_device_indices_array_buffer;
};

template <template <typename> typename DataContainer>
int LightTreeBuilder<DataContainer>::bvh_triangle_index_to_emissive_triangle_index(int bvh_triangle_index) const
{
	return m_triangle_indices[bvh_triangle_index];
}

template <template <typename> typename DataContainer>
float3 LightTreeBuilder<DataContainer>::get_triangle_vertex(unsigned int linear_emissive_triangle_index, unsigned int vertex_index, const BuilderTrianglesPayload& payload) const
{
	int emissive_triangle_index = bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
	return payload.vertices_positions[payload.triangle_vertex_indices[payload.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + vertex_index]];
}

template <template <typename> typename DataContainer>
void LightTreeBuilder<DataContainer>::build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_vertex_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
{
	auto start = std::chrono::high_resolution_clock::now();

	m_nodes.resize(emissive_triangles_primitive_indices.size() * 2 - 1);
	m_centroids.resize(emissive_triangles_primitive_indices.size());
	m_triangle_indices.resize(emissive_triangles_primitive_indices.size());
	std::iota(m_triangle_indices.begin(), m_triangle_indices.end(), 0);

	BuilderTrianglesPayload triangles_payload(emissive_triangles_primitive_indices, triangle_vertex_indices, vertices_positions, material_indices, materials);

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

template <template <typename> typename DataContainer>
void LightTreeBuilder<DataContainer>::update_node_bounds(unsigned int node_index, const BuilderTrianglesPayload& triangles_payload)
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

template <template <typename> typename DataContainer>
void LightTreeBuilder<DataContainer>::subdivide_node(unsigned int node_index, const BuilderTrianglesPayload& triangles_payload)
{
	LightTreeNode& node = m_nodes[node_index];

	int split_axis;
	float split_position;
	if (!compute_split_position(node, split_axis, split_position, triangles_payload))
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
#define SPLIT_SAH 1
#define SPLIT_SAOH 2

#define SPLIT_METHOD SPLIT_SAH

template <template <typename> typename DataContainer>
bool LightTreeBuilder<DataContainer>::compute_split_position(const LightTreeNode& node, int& out_split_axis, float& out_split_position, const BuilderTrianglesPayload& triangles_payload)
{
#if SPLIT_METHOD == SPLIT_MIDPOINT
	float3 extents = node.node_bounds.get_extents();

	int split_axis = extents.y > extents.x ? (extents.z > extents.y ? 2 : 1) : (extents.z > extents.x ? 2 : 0);
	float split_position = hippt::idx(extents, split_axis) * 0.5f + hippt::idx(node.node_bounds.mini, split_axis);
#elif SPLIT_METHOD == SPLIT_SAH
	int best_axis = -1;
	float best_position = 0.0f;
	float best_cost = 1.0e30f;

	for (int axis_index = 0; axis_index < 3; axis_index++)
	{
		unsigned int first_triangle_index = node.first_triangle_index;
		for (int i = 0; i < node.triangle_count; i++)
		{
			float3 centroid = m_centroids[bvh_triangle_index_to_emissive_triangle_index(first_triangle_index + i)];
			float sah_cost = compute_sah_cost(node, axis_index, hippt::idx(centroid, axis_index), triangles_payload);
			if (sah_cost < best_cost)
			{
				best_axis = axis_index;
				best_position = hippt::idx(centroid, axis_index);
				best_cost = sah_cost;
			}
		}
	}

	float parent_area = node.node_bounds.area();
	float parent_cost = node.triangle_count * parent_area;
	if (parent_cost < best_cost)
		return false;

	out_split_axis = best_axis;
	out_split_position = best_position;
#endif

	return true;
}

template <template <typename> typename DataContainer>
float LightTreeBuilder<DataContainer>::compute_sah_cost(const LightTreeNode& node, int axis_index, float split_position, const BuilderTrianglesPayload& triangles_payload)
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

template <template <typename> typename DataContainer>
int LightTreeBuilder<DataContainer>::partition_node_primitives(unsigned int node_index, int axis, float split_position)
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

template <template <typename> typename DataContainer>
void LightTreeBuilder<DataContainer>::to_device(HIPRTRenderData& render_data)
{
	if (m_nodes.empty())
		return;

	std::vector<LightTreeNodeDevice> nodes_device(m_nodes.size());
	for (int i = 0; i < m_nodes.size(); i++)
	{
		nodes_device[i].axis = m_nodes[i].axis;
		nodes_device[i].theta_o = m_nodes[i].theta_o;
		nodes_device[i].theta_e = m_nodes[i].theta_e;
		nodes_device[i].total_power = m_nodes[i].total_power;

		nodes_device[i].bounds_min = m_nodes[i].node_bounds.mini;
		nodes_device[i].bounds_max = m_nodes[i].node_bounds.maxi;
		nodes_device[i].left_child_index = m_nodes[i].left_child_index;
		nodes_device[i].first_triangle_index = m_nodes[i].first_triangle_index;
		nodes_device[i].triangle_count = m_nodes[i].triangle_count;
	}

	if constexpr (std::is_same<DataContainer<int>, std::vector<int>>::value)
	{
		m_device_nodes_buffer = nodes_device;
		m_device_indices_array_buffer = m_triangle_indices;

		render_data.buffers.light_tree.nodes = m_device_nodes_buffer.data();
		render_data.buffers.light_tree.indices_array = m_device_indices_array_buffer.data();
	}
	else
	{
		m_device_nodes_buffer = OrochiBuffer<LightTreeNodeDevice>(nodes_device);
		m_device_indices_array_buffer = OrochiBuffer<int>(m_triangle_indices);

		render_data.buffers.light_tree.nodes = m_device_nodes_buffer.get_device_pointer();
		render_data.buffers.light_tree.indices_array = m_device_indices_array_buffer.get_device_pointer();
	}
}

template <template <typename> typename DataContainer>
void LightTreeBuilder<DataContainer>::cleanup()
{
	m_nodes.clear();
	m_centroids.clear();
	m_triangle_indices.clear();
}

#endif
