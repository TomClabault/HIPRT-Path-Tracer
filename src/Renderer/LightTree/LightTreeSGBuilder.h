/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_H

#include "Renderer/LightTree/LightTreeATSBuilder.h"
#include "Renderer/LightTree/LightTreeBuilderCommon.h"
#include "Renderer/LightTree/LightTreeSGBuilderDeviceData.h"
#include "Renderer/LightTree/LightTreeSGNode.h"

class LightTreeSGBuilder
{
public:
	void build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials);

	void compute_node_spherical_gaussian(unsigned int node_index, const LightTreeBuilderTrianglesData& triangle_data);

	template <template <typename> typename DataContainer>
	LightTreeSGBuilderDeviceData<DataContainer> compute_device_data() const;

	template <template <typename> typename DataContainer>
	void to_device(HIPRTRenderData& render_data, const std::vector<int>& emissive_triangles_primitive_indices, unsigned int total_scene_triangle_count, LightTreeSGBuilderDeviceData<DataContainer>& device_data);

	void cleanup();

	LightTreeATSBuilderOptions& get_build_options();

private:
	LightTreeATSBuilder m_light_tree_ats_builder;

	std::vector<LightTreeSGNode> m_nodes;
};

template <template <typename> typename DataContainer>
LightTreeSGBuilderDeviceData<DataContainer> LightTreeSGBuilder::compute_device_data() const
{
	if (m_nodes.empty())
		return LightTreeSGBuilderDeviceData<DataContainer>();

	LightTreeSGBuilderDeviceData<DataContainer> device_data_out;
	device_data_out.nodes_device.resize(m_nodes.size());

	for (int i = 0; i < m_nodes.size(); i++)
	{
		device_data_out.nodes_device[i].vmf_axis = m_nodes[i].vmf_axis;
		device_data_out.nodes_device[i].vmf_sharpness = m_nodes[i].vmf_sharpness;
		device_data_out.nodes_device[i].gaussian_spatial_mean = m_nodes[i].spatial_mean;
		device_data_out.nodes_device[i].gaussian_spatial_variance = m_nodes[i].spatial_variance;
		device_data_out.nodes_device[i].bounding_sphere_radius = m_nodes[i].bounding_sphere_radius;
		device_data_out.nodes_device[i].total_power = m_nodes[i].total_power;
		device_data_out.nodes_device[i].total_emission= m_nodes[i].total_emission;
		device_data_out.nodes_device[i].bounds_min = m_nodes[i].bounds.mini;
		device_data_out.nodes_device[i].bounds_max = m_nodes[i].bounds.maxi;
		device_data_out.nodes_device[i].triangle_count = m_nodes[i].triangle_count;
		if (m_nodes[i].triangle_count == 0)
			device_data_out.nodes_device[i].left_child_index_or_first_triangle_index = m_nodes[i].left_child_index;
		else
			device_data_out.nodes_device[i].left_child_index_or_first_triangle_index = m_nodes[i].first_triangle_index;
	}

	return device_data_out;
}

template <template <typename> typename DataContainer>
void LightTreeSGBuilder::to_device(HIPRTRenderData& render_data, const std::vector<int>& emissive_triangles_primitive_indices, unsigned int total_scene_triangle_count, LightTreeSGBuilderDeviceData<DataContainer>& device_data)
{
	if (device_data.nodes_device.size() == 0)
		return;

	std::vector<unsigned int> converted_bit_trails(total_scene_triangle_count, 0xFFFFFFFF);
	for (int i = 0; i < m_light_tree_ats_builder.get_bit_trails().size(); i++)
		converted_bit_trails[emissive_triangles_primitive_indices[m_light_tree_ats_builder.get_triangle_indices()[i]]] = m_light_tree_ats_builder.get_bit_trails()[i];

	if constexpr (std::is_same_v<DataContainer<int>, std::vector<int>>)
	{
		device_data.m_device_nodes_buffer = device_data.nodes_device;
		device_data.m_device_indices_array_buffer = m_light_tree_ats_builder.get_triangle_indices();
		device_data.m_bit_trails_buffer = converted_bit_trails;
	}
	else
	{
		device_data.m_device_nodes_buffer = OrochiBuffer<LightTreeSGNodeDevice>(device_data.nodes_device);
		device_data.m_device_indices_array_buffer = OrochiBuffer<int>(m_light_tree_ats_builder.get_triangle_indices());
		device_data.m_bit_trails_buffer = OrochiBuffer<unsigned int>(converted_bit_trails);
	}

	render_data.buffers.light_tree_sg.nodes = device_data.m_device_nodes_buffer.data();
	render_data.buffers.light_tree_sg.indices_array = device_data.m_device_indices_array_buffer.data();
	render_data.buffers.light_tree_sg.bit_trails = device_data.m_bit_trails_buffer.data();
}

#endif
