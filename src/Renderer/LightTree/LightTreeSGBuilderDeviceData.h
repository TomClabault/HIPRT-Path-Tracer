/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_DEVICE_DATA_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_DEVICE_DATA_H

#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"

#include <vector>

template <template <typename> typename DataContainer>
struct LightTreeSGBuilderDeviceData
{
	void free()
	{
		if (m_device_nodes_buffer.size() == 0 && m_device_spatial_lobes_buffer.size() == 0 && m_device_tree_cut_node_indices_buffer.size() == 0 &&
			m_device_tree_cut_node_indices_neural_many_lights_buffer.size() == 0)
			return;

		if constexpr (std::is_same_v<std::vector<int>, DataContainer<int>>)
		{
			m_device_nodes_buffer									 = std::vector<LightTreeSGNodeDevice>();
			m_device_spatial_lobes_buffer							 = std::vector<SpatialSGLobeDevice>();
			m_device_tree_cut_node_indices_buffer					 = std::vector<unsigned int>();
			m_device_tree_cut_node_indices_neural_many_lights_buffer = std::vector<unsigned int>();
			m_device_indices_array_buffer							 = std::vector<int>();
			m_bit_trails_buffer										 = std::vector<unsigned int>();
		}
		else
		{
			m_device_nodes_buffer.free();
			m_device_spatial_lobes_buffer.free();
			m_device_tree_cut_node_indices_buffer.free();
			m_device_tree_cut_node_indices_neural_many_lights_buffer.free();
			m_device_indices_array_buffer.free();
			m_bit_trails_buffer.free();
		}
	}

	size_t get_VRAM_usage_bytes() const
	{
		return m_device_nodes_buffer.get_byte_size() + m_device_spatial_lobes_buffer.get_byte_size() + m_device_tree_cut_node_indices_buffer.get_byte_size() +
			   m_device_tree_cut_node_indices_neural_many_lights_buffer.get_byte_size() + m_device_indices_array_buffer.get_byte_size() +
			   m_bit_trails_buffer.get_byte_size();
	}

	std::vector<LightTreeSGNodeDevice> nodes_device;
	std::vector<SpatialSGLobeDevice> spatial_lobes_device;
	std::vector<unsigned int> tree_cut_node_indices_device;
	std::vector<unsigned int> tree_cut_node_indices_neural_many_lights_device;

	DataContainer<LightTreeSGNodeDevice> m_device_nodes_buffer;
	DataContainer<SpatialSGLobeDevice> m_device_spatial_lobes_buffer;
	DataContainer<unsigned int> m_device_tree_cut_node_indices_buffer;
	DataContainer<unsigned int> m_device_tree_cut_node_indices_neural_many_lights_buffer;
	DataContainer<int> m_device_indices_array_buffer;
	DataContainer<unsigned int> m_bit_trails_buffer;
};

#endif
