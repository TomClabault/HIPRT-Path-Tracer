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
	struct LightTreeNodeOrientationData
	{
		static constexpr float UNINITIALIZED_AXIS = -42.0f;

		void cone_union_with(float3 axis_b, float theta_o_b, float theta_e_b)
		{
			float3 axis_a = this->axis;
			float theta_o_a = this->theta_o;
			float theta_e_a = this->theta_e;

			if (axis_a.x == UNINITIALIZED_AXIS)
			{
				this->axis = axis_b;
				this->theta_o = theta_o_b;
				this->theta_e = theta_e_b;

				return;
			}

			if (theta_o_b > theta_o_a)
			{
				std::swap(theta_o_a, theta_o_b);
				std::swap(axis_a, axis_b);
				std::swap(theta_e_a, theta_e_b);
			}

			float theta_d = acos(hippt::clamp(-1.0f, 1.0f, hippt::dot(axis_a, axis_b)));
			float theta_e = hippt::max(theta_e_a, theta_e_b);

			if (hippt::min(theta_d + theta_o_b, (float)M_PI) <= theta_o_a)
			{
				this->axis = axis_a;
				this->theta_o = theta_o_a;
				this->theta_e = theta_e;

				return;
			}
			else
			{
				float theta_o = (theta_o_a + theta_d + theta_o_b) / 2.0f;
				if (M_PI <= theta_o)
				{
					this->axis = axis_a;
					this->theta_o = M_PI;
					this->theta_e = theta_e;

					return;
				}

				float theta_r = theta_o - theta_o_a;
				float3 axis = rotate_vector(axis_a, hippt::cross(axis_a, axis_b), theta_r);

				this->axis = axis;
				this->theta_o = theta_o;
				this->theta_e = theta_e;

				return;
			}
		}

		void cone_union_with(const LightTreeNodeOrientationData& other)
		{
			cone_union_with(other.axis, other.theta_o, other.theta_e);
		}

		// Axis of the cluster
		float3 axis = make_float3(UNINITIALIZED_AXIS, UNINITIALIZED_AXIS, UNINITIALIZED_AXIS);
		// Normal bounds
		float theta_o = 0.0f;
		// Emission extents
		float theta_e = 0.0f;
	};

	struct LightTreeNode
	{
		void cone_union_with(float3 other_axis, float other_theta_o, float other_theta_e)
		{
			orientation_data.cone_union_with(other_axis, other_theta_o, other_theta_e);
		}

		LightTreeNodeOrientationData orientation_data;

		// Total emissive power of the node
		ColorRGB32F total_power;

		AABB node_bounds;
		unsigned int left_child_index;
		unsigned int first_triangle_index, triangle_count;
	};

	struct Bin
	{
		AABB bounds;
		unsigned int tri_count = 0;

		// For SAOH
		LightTreeNodeOrientationData orientation_data;
		ColorRGB32F total_power;
	};

	struct BinCostInfo
	{
		float area = 0.0f;
		unsigned int tri_count = 0;

		// Needed for SAOH
		float energy = 0.0f;
		float m_omega = 0.0f;
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
	float compute_saoh_m_omega(const LightTreeNodeOrientationData& orientation_data) const;
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
		device_data_out.nodes_device[i].axis = m_nodes[i].orientation_data.axis;
		device_data_out.nodes_device[i].theta_o = m_nodes[i].orientation_data.theta_o;
		device_data_out.nodes_device[i].theta_e = m_nodes[i].orientation_data.theta_e;
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
