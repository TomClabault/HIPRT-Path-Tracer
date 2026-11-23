/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_H

#include "Device/includes/LightSampling/LightTree/SphericalGaussianUtils.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeLearningToClusterOptions.h"

#include "Renderer/LightTree/LightTreeATSBuilder.h"
#include "Renderer/LightTree/LightTreeBuilderCommon.h"
#include "Renderer/LightTree/LightTreeSGBuilderDeviceData.h"
#include "Renderer/LightTree/LightTreeSGBuilderOptions.h"
#include "Renderer/LightTree/LightTreeSGNode.h"

class LightTreeSGBuilder
{
public:
	void build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices,
						  const std::vector<float>& triangles_average_emissive_power_luminance,
						  const std::vector<int>& triangle_indices,
						  const std::vector<float3_t>& vertices_positions);

	void compute_node_spherical_gaussian(unsigned int node_index, const LightTreeBuilderTrianglesData& triangle_data);

	template <template <typename> typename DataContainer>
	LightTreeSGBuildResult<DataContainer> compute_build_result() const;

	template <template <typename> typename DataContainer>
	void to_device(HIPRTRenderData& render_data,
				   const std::vector<int>& emissive_triangles_primitive_indices,
				   unsigned int total_scene_triangle_count,
				   LightTreeSGBuildResult<DataContainer>& build_result);

	void cleanup();

	LightTreeSGBuilderOptions& get_build_options();

	int get_tree_cut_size() const;
	void set_tree_cut_size(int tree_cut_size);
	int get_second_tree_cut_size() const;
	void set_second_tree_cut_size(int second_tree_cut_size);

private:
	/**
	 * Radius of the smallest sphere centered on 'center' that contains 'bounds'
	 */
	static float bounds_sphere_radius(const AABB& bounds, float3_t center);
	void update_ats_builder_options();
	void compute_tree_cut(std::vector<unsigned int>& tree_cut_node_indices, unsigned int& effective_tree_cut_size, int tree_cut_size);
	template <template <typename> typename DataContainer>
	LightTreeSGBuilderDeviceData<DataContainer> compute_device_data() const;

private:
	LightTreeATSBuilder m_light_tree_ats_builder;

	std::vector<LightTreeSGNode> m_nodes;

	std::vector<unsigned int> m_tree_cut_node_indices;
	unsigned int m_effective_tree_cut_size = 0;

	std::vector<unsigned int> m_second_tree_cut_node_indices;
	unsigned int m_effective_second_tree_cut_size = 0;
	int m_second_tree_cut_size					  = LearningToClusterInitialLightCutSize;

	LightTreeSGBuilderOptions m_build_options;
};

template <template <typename> typename DataContainer>
LightTreeSGBuildResult<DataContainer> LightTreeSGBuilder::compute_build_result() const
{
	LightTreeSGBuildResult<DataContainer> build_result;
	build_result.device_data					= compute_device_data<DataContainer>();
	build_result.tree_cut_node_indices			= m_tree_cut_node_indices;
	build_result.effective_tree_cut_size		= m_effective_tree_cut_size;
	build_result.second_tree_cut_node_indices	= m_second_tree_cut_node_indices;
	build_result.effective_second_tree_cut_size = m_effective_second_tree_cut_size;

	return build_result;
}

template <template <typename> typename DataContainer>
LightTreeSGBuilderDeviceData<DataContainer> LightTreeSGBuilder::compute_device_data() const
{
	if (m_nodes.empty())
		return LightTreeSGBuilderDeviceData<DataContainer>();

	LightTreeSGBuilderDeviceData<DataContainer> device_data_out;
	device_data_out.nodes_device.resize(m_nodes.size());
	device_data_out.node_energy_statistics_device.resize(m_nodes.size());
	device_data_out.tree_cut_node_indices_device		= m_tree_cut_node_indices;
	device_data_out.second_tree_cut_node_indices_device = m_second_tree_cut_node_indices;

	for (int i = 0; i < m_nodes.size(); i++)
	{
		const LightTreeSGNode& node		   = m_nodes[i];
		LightTreeSGNodeDevice& device_node = device_data_out.nodes_device[i];

		device_node.spatial_mean		   = node.get_float_spatial_mean();
		device_node.total_power			   = static_cast<float>(node.total_power / SG_integral(node.vmf.sharpness));
		device_node.vmf.axis			   = node.vmf.axis;
		device_node.vmf.sharpness		   = node.vmf.sharpness;
		device_node.orientation_axis	   = node.orientation_axis;
		device_node.bounding_sphere_radius = node.bounding_sphere_radius;
		device_node.bounds_min			   = node.bounds.mini;
		device_node.bounds_max			   = node.bounds.maxi;
		device_node.cos_theta_o			   = cosf(node.theta_o);
		device_node.sin_theta_o			   = sinf(node.theta_o);
		device_node.spatial_variance	   = static_cast<float>(node.spatial_variance);
		device_node.support_radius		   = bounds_sphere_radius(node.bounds, device_node.spatial_mean);
		device_node.triangle_count		   = node.triangle_count;

		if (node.triangle_count == 0)
			device_node.left_child_index_or_first_triangle_index = node.left_child_index;
		else
			device_node.left_child_index_or_first_triangle_index = node.first_triangle_index;

		LightTreeSGNodeEnergyStatisticsDevice& energy_statistics = device_data_out.node_energy_statistics_device[i];
		energy_statistics.energy_variance						 = node.energy_variance;
		energy_statistics.energy_average						 = node.energy_average;
		energy_statistics.total_emitter_count					 = node.total_emitter_count;
	}

	// Path from the root to every node, with the encoding of the bit trails of the triangles: the device finds the node of a light cut that
	// contains a triangle by comparing these paths with the bit trail of the triangle. Nodes deeper than 32 keep the invalid depth: their path
	// does not fit in a bit trail
	device_data_out.node_path_codes_device.resize(m_nodes.size(), make_uint2(0u, LightTreeSGDevice::INVALID_NODE_PATH_DEPTH));
	device_data_out.node_path_codes_device[0] = make_uint2(0u, 0u);

	std::vector<unsigned int> nodes_to_visit = { 0u };
	while (!nodes_to_visit.empty())
	{
		unsigned int node_index = nodes_to_visit.back();
		nodes_to_visit.pop_back();

		uint2_t node_path_code = device_data_out.node_path_codes_device[node_index];
		if (m_nodes[node_index].triangle_count != 0 || node_path_code.y >= 32u)
			// Leaf, or children too deep for a 32-bit trail
			continue;

		unsigned int left_child_index  = m_nodes[node_index].left_child_index;
		unsigned int right_child_index = left_child_index + 1;

		device_data_out.node_path_codes_device[left_child_index]  = make_uint2(node_path_code.x, node_path_code.y + 1u);
		device_data_out.node_path_codes_device[right_child_index] = make_uint2(node_path_code.x | (1u << node_path_code.y), node_path_code.y + 1u);

		nodes_to_visit.push_back(left_child_index);
		nodes_to_visit.push_back(right_child_index);
	}

	return device_data_out;
}

template <template <typename> typename DataContainer>
void LightTreeSGBuilder::to_device(HIPRTRenderData& render_data,
								   const std::vector<int>& emissive_triangles_primitive_indices,
								   unsigned int total_scene_triangle_count,
								   LightTreeSGBuildResult<DataContainer>& build_result)
{
	LightTreeSGBuilderDeviceData<DataContainer>& device_data = build_result.device_data;
	if (device_data.nodes_device.size() == 0)
	{
		render_data.light_tree_sg.settings.effective_tree_cut_size		  = 0;
		render_data.light_tree_sg.settings.effective_second_tree_cut_size = 0;
		render_data.light_tree_sg.nodes									  = nullptr;
		render_data.light_tree_sg.node_energy_statistics				  = nullptr;
		render_data.light_tree_sg.tree_cut_node_indices					  = nullptr;
		render_data.light_tree_sg.second_tree_cut_node_indices			  = nullptr;
		render_data.light_tree_sg.node_path_codes						  = nullptr;

		return;
	}

	std::vector<unsigned int> converted_bit_trails(total_scene_triangle_count, 0xFFFFFFFF);

	for (int i = 0; i < m_light_tree_ats_builder.get_bit_trails().size(); i++)
		converted_bit_trails[emissive_triangles_primitive_indices[m_light_tree_ats_builder.get_triangle_indices()[i]]] =
			m_light_tree_ats_builder.get_bit_trails()[i];

	if constexpr (std::is_same_v<DataContainer<int>, std::vector<int>>)
	{
		device_data.m_device_node_energy_statistics_buffer		 = device_data.node_energy_statistics_device;
		device_data.m_device_tree_cut_node_indices_buffer		 = device_data.tree_cut_node_indices_device;
		device_data.m_device_second_tree_cut_node_indices_buffer = device_data.second_tree_cut_node_indices_device;

		device_data.m_device_nodes_buffer			= device_data.nodes_device;
		device_data.m_device_indices_array_buffer	= m_light_tree_ats_builder.get_triangle_indices();
		device_data.m_bit_trails_buffer				= converted_bit_trails;
		device_data.m_device_node_path_codes_buffer = device_data.node_path_codes_device;
	}
	else
	{
		device_data.m_device_node_energy_statistics_buffer = OrochiBuffer<LightTreeSGNodeEnergyStatisticsDevice>(device_data.node_energy_statistics_device);
		device_data.m_device_tree_cut_node_indices_buffer  = OrochiBuffer<unsigned int>(device_data.tree_cut_node_indices_device);
		device_data.m_device_second_tree_cut_node_indices_buffer = OrochiBuffer<unsigned int>(device_data.second_tree_cut_node_indices_device);

		device_data.m_device_nodes_buffer			= OrochiBuffer<LightTreeSGNodeDevice>(device_data.nodes_device);
		device_data.m_device_indices_array_buffer	= OrochiBuffer<int>(m_light_tree_ats_builder.get_triangle_indices());
		device_data.m_bit_trails_buffer				= OrochiBuffer<unsigned int>(converted_bit_trails);
		device_data.m_device_node_path_codes_buffer = OrochiBuffer<uint2_t>(device_data.node_path_codes_device);
	}

	render_data.light_tree_sg.settings.effective_tree_cut_size		  = build_result.effective_tree_cut_size;
	render_data.light_tree_sg.settings.effective_second_tree_cut_size = build_result.effective_second_tree_cut_size;
	render_data.light_tree_sg.nodes									  = device_data.m_device_nodes_buffer.data();
	render_data.light_tree_sg.node_energy_statistics				  = device_data.m_device_node_energy_statistics_buffer.data();
	render_data.light_tree_sg.tree_cut_node_indices					  = device_data.m_device_tree_cut_node_indices_buffer.data();
	render_data.light_tree_sg.second_tree_cut_node_indices			  = device_data.m_device_second_tree_cut_node_indices_buffer.data();
	render_data.light_tree_sg.indices_array							  = device_data.m_device_indices_array_buffer.data();
	render_data.light_tree_sg.bit_trails							  = device_data.m_bit_trails_buffer.data();
	render_data.light_tree_sg.node_path_codes						  = device_data.m_device_node_path_codes_buffer.data();
}

#endif // #ifndef RENDERER_LIGHT_TREE_SG_BUILDER_H
