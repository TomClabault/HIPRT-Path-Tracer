/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_BUILDER_DEVICE_DATA_H
#define RENDERER_LIGHT_TREE_BUILDER_DEVICE_DATA_H

#include "Device/includes/LightSampling/LightTree/LightTreeDevice.h"

#include <vector>

template <template <typename> typename DataContainer>
struct LightTreeBuilderDeviceData
{
	void free()
	{
		if (m_device_nodes_buffer.size() == 0)
			return;

		if constexpr (std::is_same_v<std::vector<int>, DataContainer<int>>)
		{
			m_device_nodes_buffer = std::vector<LightTreeNodeDevice>();
			m_device_indices_array_buffer = std::vector<int>();
		}
		else
		{
			m_device_nodes_buffer.free();
			m_device_indices_array_buffer.free();
		}
	}

	std::vector<LightTreeNodeDevice> nodes_device;

	DataContainer<LightTreeNodeDevice> m_device_nodes_buffer;
	DataContainer<int> m_device_indices_array_buffer;
	DataContainer<unsigned int> m_bit_trails_buffer;
};

#endif
