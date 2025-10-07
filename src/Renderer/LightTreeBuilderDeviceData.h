/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_BUILDER_DEVICE_DATA_H
#define RENDERER_LIGHT_TREE_BUILDER_DEVICE_DATA_H

#include "Device/includes/LightSampling/LightTreeDevice.h"

#include <vector>

template <template <typename> typename DataContainer>
struct LightTreeBuilderDeviceData
{
	std::vector<LightTreeNodeDevice> nodes_device;

	DataContainer<LightTreeNodeDevice> m_device_nodes_buffer;
	DataContainer<int> m_device_indices_array_buffer;
};

#endif
