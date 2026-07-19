/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_H

#include "Device/includes/LightSampling/LightTree/SphericalGaussianUtils.h"

#include "Renderer/LightTree/LightTreeATSBuilder.h"
#include "Renderer/LightTree/LightTreeBuilderCommon.h"
#include "Renderer/LightTree/LightTreeSGBuilderDeviceData.h"
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
	LightTreeSGBuilderDeviceData<DataContainer> compute_device_data() const;

	template <template <typename> typename DataContainer>
	void to_device(HIPRTRenderData& render_data,
				   const std::vector<int>& emissive_triangles_primitive_indices,
				   unsigned int total_scene_triangle_count,
				   LightTreeSGBuilderDeviceData<DataContainer>& device_data);

	void cleanup();

	LightTreeATSBuilderOptions& get_build_options();
	int get_spatial_lobe_count() const;
	void set_spatial_lobe_count(int spatial_lobe_count);

private:
	LightTreeATSBuilder m_light_tree_ats_builder;

	std::vector<LightTreeSGNode> m_nodes;
	int m_spatial_lobe_count = 2;
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
		device_data_out.nodes_device[i].vmf.axis			  = m_nodes[i].vmf.axis;
		device_data_out.nodes_device[i].vmf.sharpness		  = m_nodes[i].vmf.sharpness;
		device_data_out.nodes_device[i].gaussian_spatial_mean = m_nodes[i].spatial_mean;
		device_data_out.nodes_device[i].spatial_lobe_count	  = m_spatial_lobe_count;
		for (int lobe_index = 0; lobe_index < LIGHT_TREE_SG_MAX_SPATIAL_LOBES; lobe_index++)
		{
			const LightTreeSGSpatialLobeBuild& lobe = m_nodes[i].spatial_lobes[lobe_index];
			SpatialSGLobeDevice& device_lobe		= device_data_out.nodes_device[i].spatial_lobes[lobe_index];
			device_lobe.mean	 = make_float3(static_cast<float>(lobe.mean_x), static_cast<float>(lobe.mean_y), static_cast<float>(lobe.mean_z));
			device_lobe.variance = static_cast<float>(lobe.variance);
			device_lobe.power	 = static_cast<float>(lobe.power / SG_integral(m_nodes[i].vmf.sharpness));

			float radius_squared = 0.0f;
			for (int corner_index = 0; corner_index < 8; corner_index++)
			{
				const float3_t corner =
					make_float3((corner_index & 1) ? lobe.bounds.mini.x : lobe.bounds.maxi.x, (corner_index & 2) ? lobe.bounds.mini.y : lobe.bounds.maxi.y,
								(corner_index & 4) ? lobe.bounds.mini.z : lobe.bounds.maxi.z);
				radius_squared = hippt::max(radius_squared, hippt::length2(corner - device_lobe.mean));
			}
			device_lobe.support_radius = hippt::sqrt(radius_squared);
		}
		device_data_out.nodes_device[i].orientation_axis	   = m_nodes[i].orientation_axis;
		device_data_out.nodes_device[i].cos_theta_o			   = cosf(m_nodes[i].theta_o);
		device_data_out.nodes_device[i].sin_theta_o			   = sinf(m_nodes[i].theta_o);
		device_data_out.nodes_device[i].bounding_sphere_radius = m_nodes[i].bounding_sphere_radius;
		device_data_out.nodes_device[i].total_power			   = m_nodes[i].total_power / SG_integral(m_nodes[i].vmf.sharpness);
		device_data_out.nodes_device[i].energy_variance		   = m_nodes[i].energy_variance;
		device_data_out.nodes_device[i].energy_average		   = m_nodes[i].energy_average;
		device_data_out.nodes_device[i].total_emitter_count	   = m_nodes[i].total_emitter_count;
		device_data_out.nodes_device[i].bounds_min			   = m_nodes[i].bounds.mini;
		device_data_out.nodes_device[i].bounds_max			   = m_nodes[i].bounds.maxi;
		device_data_out.nodes_device[i].triangle_count		   = m_nodes[i].triangle_count;
		if (m_nodes[i].triangle_count == 0)
			device_data_out.nodes_device[i].left_child_index_or_first_triangle_index = m_nodes[i].left_child_index;
		else
			device_data_out.nodes_device[i].left_child_index_or_first_triangle_index = m_nodes[i].first_triangle_index;
	}

	return device_data_out;
}

template <template <typename> typename DataContainer>
void LightTreeSGBuilder::to_device(HIPRTRenderData& render_data,
								   const std::vector<int>& emissive_triangles_primitive_indices,
								   unsigned int total_scene_triangle_count,
								   LightTreeSGBuilderDeviceData<DataContainer>& device_data)
{
	if (device_data.nodes_device.size() == 0)
		return;

	std::vector<unsigned int> converted_bit_trails(total_scene_triangle_count, 0xFFFFFFFF);
	for (int i = 0; i < m_light_tree_ats_builder.get_bit_trails().size(); i++)
		converted_bit_trails[emissive_triangles_primitive_indices[m_light_tree_ats_builder.get_triangle_indices()[i]]] =
			m_light_tree_ats_builder.get_bit_trails()[i];

	if constexpr (std::is_same_v<DataContainer<int>, std::vector<int>>)
	{
		device_data.m_device_nodes_buffer		  = device_data.nodes_device;
		device_data.m_device_indices_array_buffer = m_light_tree_ats_builder.get_triangle_indices();
		device_data.m_bit_trails_buffer			  = converted_bit_trails;
	}
	else
	{
		device_data.m_device_nodes_buffer		  = OrochiBuffer<LightTreeSGNodeDevice>(device_data.nodes_device);
		device_data.m_device_indices_array_buffer = OrochiBuffer<int>(m_light_tree_ats_builder.get_triangle_indices());
		device_data.m_bit_trails_buffer			  = OrochiBuffer<unsigned int>(converted_bit_trails);
	}

	render_data.light_tree_sg.nodes			= device_data.m_device_nodes_buffer.data();
	render_data.light_tree_sg.indices_array = device_data.m_device_indices_array_buffer.data();
	render_data.light_tree_sg.bit_trails	= device_data.m_bit_trails_buffer.data();
}

#endif
