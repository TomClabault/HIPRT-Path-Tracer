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
#include "Renderer/LightTreeNodeOrientationData.h"
#include "Scene/AABB.h"

class LightTreeBuilder
{
public:
	struct PrefetchedTriangle
	{
		AABB bounds;

		float3 centroid;
		float3 normal;
		float area;

		ColorRGB32F power;
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
		unsigned int right_child_index;
		unsigned int first_triangle_index, triangle_count;
		unsigned int bit_trail = 0;
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
	void subdivide_node(unsigned int node_index, const BuilderTrianglesData& triangles_data, int depth);
	float compute_saoh_m_omega(const LightTreeNodeOrientationData& orientation_data) const;
	float compute_split_position(const LightTreeNode& node, int& out_split_axis, float& out_split_position, const BuilderTrianglesData& triangles_data);
	float compute_node_cost(const LightTreeNode& node);
	float compute_sah_cost(const LightTreeNode& node, int axis_index, float split_position, const BuilderTrianglesData& triangles_data);
	int partition_node_primitives(unsigned int node_index, int axis, float split_position);
	void register_node_bit_trail(const LightTreeNode& node, const BuilderTrianglesData& triangles_data);

	template <template <typename> typename DataContainer>
	LightTreeBuilderDeviceData<DataContainer> compute_device_data() const;

	template <template <typename> typename DataContainer>
	void to_device(HIPRTRenderData& render_data, const std::vector<int>& emissive_triangles_primitive_indices, unsigned int total_scene_triangle_count, LightTreeBuilderDeviceData<DataContainer>& device_data);

	/**
	 * Frees up the memory that was needed for building the tree
	 */
	void cleanup();

	LightTreeBuilderOptions& get_options();

private:
	LightTreeBuilderOptions m_build_options;

	std::shared_ptr<std::atomic<unsigned int>> m_current_node_index = 0;
	std::vector<LightTreeNode> m_nodes;

	std::vector<PrefetchedTriangle> m_prefetched_triangles;
	std::vector<int> m_triangle_indices; // Indices of the emissive triangles from 0 to N - 1
	std::vector<unsigned int> m_bit_trails; // Indices of the emissive triangles from 0 to N - 1
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
		device_data_out.nodes_device[i].right_child_index = m_nodes[i].right_child_index;
		device_data_out.nodes_device[i].first_triangle_index = m_nodes[i].first_triangle_index;
		device_data_out.nodes_device[i].triangle_count = m_nodes[i].triangle_count;
	}

	return device_data_out;
}

template <template <typename> typename DataContainer>
void LightTreeBuilder::to_device(HIPRTRenderData& render_data, const std::vector<int>& emissive_triangles_primitive_indices, unsigned int total_scene_triangle_count, LightTreeBuilderDeviceData<DataContainer>& device_data)
{
	if (device_data.nodes_device.size() == 0)
		return;

	std::vector<unsigned int> converted_bit_trails(total_scene_triangle_count, 0xFFFFFFFF);
	for (int i = 0; i < m_bit_trails.size(); i++)
		converted_bit_trails[emissive_triangles_primitive_indices[m_triangle_indices[i]]] = m_bit_trails[i];

	if constexpr (std::is_same<DataContainer<int>, std::vector<int>>::value)
	{
		device_data.m_device_nodes_buffer = device_data.nodes_device;
		device_data.m_device_indices_array_buffer = m_triangle_indices;
		device_data.m_bit_trails_buffer = converted_bit_trails;
	}
	else
	{
		device_data.m_device_nodes_buffer = OrochiBuffer<LightTreeNodeDevice>(device_data.nodes_device);
		device_data.m_device_indices_array_buffer = OrochiBuffer<int>(m_triangle_indices);
		device_data.m_bit_trails_buffer = OrochiBuffer<unsigned int>(converted_bit_trails);
	}

	render_data.buffers.light_tree.nodes = device_data.m_device_nodes_buffer.data();
	render_data.buffers.light_tree.indices_array = device_data.m_device_indices_array_buffer.data();
	render_data.buffers.light_tree.bit_trails = device_data.m_bit_trails_buffer.data();
}

#endif
