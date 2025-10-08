/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_BUILDER_H
#define RENDERER_LIGHT_TREE_BUILDER_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Material/MaterialCPU.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/LightTreeBuilderDeviceData.h"
#include "Renderer/LightTreeBuilderOptions.h"
#include "Scene/AABB.h"

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

			float theta_d = acos(hippt::clamp(-1.0f, 1.0f, hippt::dot(axis_a, axis_b)));
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

	struct Bin
	{
		AABB bounds;
		unsigned int tri_count = 0;
	};

	struct BuilderTrianglesData
	{
		BuilderTrianglesData(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_vertex_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
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

	float3 get_triangle_vertex(unsigned int linear_emissive_triangle_index, unsigned int vertex_index, const BuilderTrianglesData& triangles_data) const;

	void build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials);

	void update_node_bounds(unsigned int node_index, const BuilderTrianglesData& triangles_data);
	void subdivide_node(unsigned int node_index, const BuilderTrianglesData& triangles_data);
	float compute_split_position(const LightTreeNode& node, int& out_split_axis, float& out_split_position, const BuilderTrianglesData& triangles_data);
	float compute_node_cost(const LightTreeNode& node);
	float compute_sah_cost(const LightTreeNode& node, int axis_index, float split_position, const BuilderTrianglesData& triangles_data);
	int partition_node_primitives(unsigned int node_index, int axis, float split_position);

	template <template <typename> typename DataContainer>
	LightTreeBuilderDeviceData<DataContainer> compute_device_data() const;

	template <template <typename> typename DataContainer>
	void to_device(HIPRTRenderData& render_data, LightTreeBuilderDeviceData<DataContainer>& device_data);

	/**
	 * Frees up the memory that was needed for building the tree
	 */
	void cleanup();

	LightTreeBuilderOptions& get_options();

private:
	LightTreeBuilderOptions m_build_options;

	unsigned int m_current_node_index = 0;
	std::vector<LightTreeNode> m_nodes;

	std::vector<float3> m_centroids;
	std::vector<int> m_triangle_indices; // Indices of the emissive triangles from 0 to N - 1
};

template <template <typename> typename DataContainer>
LightTreeBuilderDeviceData<DataContainer> LightTreeBuilder::compute_device_data() const
{
	if (m_nodes.empty())
		return LightTreeBuilderDeviceData<DataContainer>();

	LightTreeBuilderDeviceData<DataContainer> device_data_out;
	device_data_out.nodes_device.resize(m_nodes.size());

	for (int i = 0; i < m_nodes.size(); i++)
	{
		device_data_out.nodes_device[i].axis = m_nodes[i].axis;
		device_data_out.nodes_device[i].theta_o = m_nodes[i].theta_o;
		device_data_out.nodes_device[i].theta_e = m_nodes[i].theta_e;
		device_data_out.nodes_device[i].total_power = m_nodes[i].total_power;

		device_data_out.nodes_device[i].bounds_min = m_nodes[i].node_bounds.mini;
		device_data_out.nodes_device[i].bounds_max = m_nodes[i].node_bounds.maxi;
		device_data_out.nodes_device[i].left_child_index = m_nodes[i].left_child_index;
		device_data_out.nodes_device[i].first_triangle_index = m_nodes[i].first_triangle_index;
		device_data_out.nodes_device[i].triangle_count = m_nodes[i].triangle_count;
	}

	return device_data_out;
}

template <template <typename> typename DataContainer>
void LightTreeBuilder::to_device(HIPRTRenderData& render_data, LightTreeBuilderDeviceData<DataContainer>& device_data)
{
	if (device_data.nodes_device.size() == 0)
		return;

	if constexpr (std::is_same<DataContainer<int>, std::vector<int>>::value)
	{
		device_data.m_device_nodes_buffer = device_data.nodes_device;
		device_data.m_device_indices_array_buffer = m_triangle_indices;

		render_data.buffers.light_tree.nodes = device_data.m_device_nodes_buffer.data();
		render_data.buffers.light_tree.indices_array = device_data.m_device_indices_array_buffer.data();
	}
	else
	{
		device_data.m_device_nodes_buffer = OrochiBuffer<LightTreeNodeDevice>(device_data.nodes_device);
		device_data.m_device_indices_array_buffer = OrochiBuffer<int>(m_triangle_indices);

		render_data.buffers.light_tree.nodes = device_data.m_device_nodes_buffer.get_device_pointer();
		render_data.buffers.light_tree.indices_array = device_data.m_device_indices_array_buffer.get_device_pointer();
	}
}

#endif
