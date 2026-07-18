/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_ATS_BUILDER_H
#define RENDERER_LIGHT_TREE_ATS_BUILDER_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Material/MaterialCPU.h"
#include "HostDeviceCommon/RenderData.h"
#include "Renderer/LightTree/LightTreeATSBuilderDeviceData.h"
#include "Renderer/LightTree/LightTreeATSBuilderOptions.h"
#include "Renderer/LightTree/LightTreeATSNode.h"
#include "Renderer/LightTree/LightTreeBuilderCommon.h"
#include "Scene/AABB.h"

class LightTreeATSBuilder
{
public:
	struct PrefetchedTriangle
	{
		AABB bounds;

		float3_t centroid;
		float3_t normal;
		float area;

		float power;
	};

	struct Bin
	{
		AABB bounds;
		unsigned int tri_count = 0;

		// For SAOH
		LightTreeATSNodeOrientationData orientation_data;
		float total_power = 0.0f;
	};

	struct BinCostInfo
	{
		float surface_area	   = 0.0f;
		unsigned int tri_count = 0;

		// Needed for SAOH
		float energy  = 0.0f;
		float m_omega = 0.0f;
	};

	int bvh_triangle_index_to_emissive_triangle_index(int bvh_triangle_index) const;

	float3_t get_triangle_vertex(unsigned int linear_emissive_triangle_index,
								 unsigned int vertex_index,
								 const LightTreeBuilderTrianglesData& triangles_data) const;

	void build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices,
						  const std::vector<float>& triangles_average_emissive_power_luminance,
						  const std::vector<int>& triangle_indices,
						  const std::vector<float3_t>& vertices_positions);

	void update_node_bounds(unsigned int node_index, const LightTreeBuilderTrianglesData& triangles_data);
	void subdivide_node(unsigned int node_index, const LightTreeBuilderTrianglesData& triangles_data, int depth);
	float compute_saoh_m_omega(const LightTreeATSNodeOrientationData& orientation_data) const;
	float compute_split_position(
		const LightTreeATSNode& node, int& out_split_axis, float& out_split_position, const LightTreeBuilderTrianglesData& triangles_data, int split_method);
	float compute_node_cost(const LightTreeATSNode& node);
	float compute_sah_cost(const LightTreeATSNode& node, int axis_index, float split_position, const LightTreeBuilderTrianglesData& triangles_data);
	int partition_node_primitives(unsigned int node_index, int axis, float split_position);
	void register_node_bit_trail(const LightTreeATSNode& node, const LightTreeBuilderTrianglesData& triangles_data);

	template <template <typename> typename DataContainer>
	LightTreeATSBuilderDeviceData<DataContainer> compute_device_data() const;

	template <template <typename> typename DataContainer>
	void to_device(HIPRTRenderData& render_data,
				   const std::vector<int>& emissive_triangles_primitive_indices,
				   unsigned int total_scene_triangle_count,
				   LightTreeATSBuilderDeviceData<DataContainer>& device_data);

	/**
	 * Frees up the memory that was needed for building the tree
	 */
	void cleanup();

	const std::vector<LightTreeATSNode>& get_nodes() const;
	const std::vector<PrefetchedTriangle>& get_prefetched_triangles() const;
	const std::vector<unsigned int>& get_bit_trails() const;
	const std::vector<int>& get_triangle_indices() const;

	LightTreeATSBuilderOptions& get_build_options();

private:
	LightTreeATSBuilderOptions m_build_options;

	std::shared_ptr<std::atomic<unsigned int>> m_max_tree_depth		= nullptr;
	std::shared_ptr<std::atomic<unsigned int>> m_current_node_index = nullptr;
	std::vector<LightTreeATSNode> m_nodes;

	std::vector<PrefetchedTriangle> m_prefetched_triangles;
	std::vector<int> m_triangle_indices;	// Original emissive-list indices of non-degenerate triangles
	std::vector<unsigned int> m_bit_trails; // Indices of the emissive triangles from 0 to N - 1
};

template <template <typename> typename DataContainer>
LightTreeATSBuilderDeviceData<DataContainer> LightTreeATSBuilder::compute_device_data() const
{
	if (m_nodes.empty())
		return LightTreeATSBuilderDeviceData<DataContainer>();

	LightTreeATSBuilderDeviceData<DataContainer> device_data_out;
	device_data_out.nodes_device.resize(m_nodes.size());

	for (int i = 0; i < m_nodes.size(); i++)
	{
		if (hippt::is_nan(m_nodes[i].orientation_data.theta_o) || hippt::is_inf(m_nodes[i].orientation_data.theta_o) ||
			hippt::is_nan(m_nodes[i].orientation_data.axis.x) || hippt::is_nan(m_nodes[i].orientation_data.axis.y) ||
			hippt::is_nan(m_nodes[i].orientation_data.axis.z) || hippt::is_inf(m_nodes[i].orientation_data.axis.x) ||
			hippt::is_inf(m_nodes[i].orientation_data.axis.y) || hippt::is_inf(m_nodes[i].orientation_data.axis.z))
			Debug::debugbreak();

		device_data_out.nodes_device[i].axis				  = m_nodes[i].orientation_data.axis;
		device_data_out.nodes_device[i].cos_theta_o			  = cosf(m_nodes[i].orientation_data.theta_o);
		device_data_out.nodes_device[i].sin_theta_o			  = sinf(m_nodes[i].orientation_data.theta_o);
		device_data_out.nodes_device[i].total_power_luminance = m_nodes[i].total_power;
		device_data_out.nodes_device[i].total_emitter_count	  = m_nodes[i].total_emitter_count;
		device_data_out.nodes_device[i].bounds_min			  = m_nodes[i].node_bounds.mini;
		device_data_out.nodes_device[i].bounds_max			  = m_nodes[i].node_bounds.maxi;
		device_data_out.nodes_device[i].triangle_count		  = m_nodes[i].triangle_count;
		if (m_nodes[i].triangle_count == 0)
			device_data_out.nodes_device[i].left_child_index_or_first_triangle_index = m_nodes[i].left_child_index;
		else
			device_data_out.nodes_device[i].left_child_index_or_first_triangle_index = m_nodes[i].first_triangle_index;
	}

	return device_data_out;
}

template <template <typename> typename DataContainer>
void LightTreeATSBuilder::to_device(HIPRTRenderData& render_data,
									const std::vector<int>& emissive_triangles_primitive_indices,
									unsigned int total_scene_triangle_count,
									LightTreeATSBuilderDeviceData<DataContainer>& device_data)
{
	if (device_data.nodes_device.size() == 0)
		return;

	std::vector<unsigned int> converted_bit_trails(total_scene_triangle_count, 0xFFFFFFFF);
	for (int i = 0; i < m_bit_trails.size(); i++)
		converted_bit_trails[emissive_triangles_primitive_indices[m_triangle_indices[i]]] = m_bit_trails[i];

	if constexpr (std::is_same<DataContainer<int>, std::vector<int>>::value)
	{
		device_data.m_device_nodes_buffer		  = device_data.nodes_device;
		device_data.m_device_indices_array_buffer = m_triangle_indices;
		device_data.m_bit_trails_buffer			  = converted_bit_trails;
	}
	else
	{
		device_data.m_device_nodes_buffer		  = OrochiBuffer<LightTreeATSNodeDevice>(device_data.nodes_device);
		device_data.m_device_indices_array_buffer = OrochiBuffer<int>(m_triangle_indices);
		device_data.m_bit_trails_buffer			  = OrochiBuffer<unsigned int>(converted_bit_trails);
	}

	render_data.light_tree_ats.nodes		 = device_data.m_device_nodes_buffer.data();
	render_data.light_tree_ats.indices_array = device_data.m_device_indices_array_buffer.data();
	render_data.light_tree_ats.bit_trails	 = device_data.m_bit_trails_buffer.data();
}

#endif
