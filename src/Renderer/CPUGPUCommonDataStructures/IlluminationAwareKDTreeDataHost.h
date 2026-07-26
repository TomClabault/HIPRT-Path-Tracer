/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H

#include "HostDeviceCommon/IlluminationAwareKDTreeDevice.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using IlluminationAwareKDTreeDataHostInternal = GenericSoA<DataContainer, IlluminationAwareKDTreeNode, IlluminationAwareKDTreeNodeBounds>;

enum IlluminationAwareKDTreeDataHostBuffers
{
	ILLUMINATION_AWARE_KD_TREE_NODES,
	ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS
};

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	void resize(uint32_t new_node_capacity)
	{
		m_nodes_and_bounds.resize(new_node_capacity);

		GenericSoAHelpers::resize<DataContainer>(m_node_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_node_count, 1);

		reset();
	}

	void reset()
	{
		if (maximum_size() == 0)
			return;

		GenericSoAHelpers::memset_buffer<DataContainer>(m_node_count, 0u);
		GenericSoAHelpers::memset_buffer<DataContainer>(m_active_guiding_node_count, 0u);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_nodes_and_bounds.free();
		m_node_count				= DataContainer<unsigned int>();
		m_active_guiding_nodes		= DataContainer<unsigned int>();
		m_active_guiding_node_count = DataContainer<unsigned int>();

		return true;
	}

	std::size_t get_byte_size() const
	{
		return m_nodes_and_bounds.get_byte_size() + GenericSoAHelpers::get_byte_size(m_node_count) + GenericSoAHelpers::get_byte_size(m_active_guiding_nodes) +
			   GenericSoAHelpers::get_byte_size(m_active_guiding_node_count);
	}

	std::size_t maximum_size() const
	{
		return m_nodes_and_bounds.maximum_size();
	}

	IlluminationAwareKDTreeDevice to_device()
	{
		IlluminationAwareKDTreeDevice device;

		device.nodes					 = m_nodes_and_bounds.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_NODES>();
		device.node_bounds				 = m_nodes_and_bounds.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>();
		device.node_count				 = m_node_count.data();
		device.node_capacity			 = static_cast<uint32_t>(maximum_size());
		device.active_guiding_nodes		 = m_active_guiding_nodes.data();
		device.active_guiding_node_count = m_active_guiding_node_count.data();

		return device;
	}

	IlluminationAwareKDTreeDataHostInternal<DataContainer> m_nodes_and_bounds;
	DataContainer<uint32_t> m_node_count;
	DataContainer<uint32_t> m_active_guiding_nodes;
	DataContainer<uint32_t> m_active_guiding_node_count;
};

#endif
