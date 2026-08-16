/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeCoreDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeNISMLDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	static constexpr unsigned int MAXIMUM_NUMBER_OF_NODES = IlluminationAwareKDTreeCoreDataHost<DataContainer>::MAXIMUM_NUMBER_OF_NODES;

	void resize(unsigned int new_node_capacity,
				unsigned int new_training_sample_capacity,
				unsigned int new_nisml_representative_capacity	 = 1,
				unsigned int new_nisml_hash_table_reserved_bytes = 100000000u,
				unsigned int new_nisml_hash_normal_precision	 = 2u)
	{
		m_kd_tree_data.resize(new_node_capacity, new_training_sample_capacity);
		m_nisml_data.resize(new_node_capacity, new_nisml_representative_capacity, new_nisml_hash_table_reserved_bytes, new_nisml_hash_normal_precision);
		m_counter_download_buffer.resize_host_pinned_mem(1);
	}

	void reset()
	{
		// All reset is already done by the render pass
	}

	bool free()
	{
		bool core_data_freed			   = m_kd_tree_data.free();
		bool nisml_data_freed			   = m_nisml_data.free();
		bool counter_download_buffer_freed = m_counter_download_buffer.maximum_size() > 0;
		if (counter_download_buffer_freed)
			m_counter_download_buffer.free();

		return core_data_freed || nisml_data_freed || counter_download_buffer_freed;
	}

	template <typename CounterBuffer>
	unsigned int download_counter(const CounterBuffer& counter)
	{
		unsigned int* host_pinned_pointer = m_counter_download_buffer.template get_host_pinned_pointer<0>();
		GenericSoAHelpers::download_data_into(counter, host_pinned_pointer);

		return host_pinned_pointer[0];
	}

	std::size_t maximum_size() const
	{
		return m_kd_tree_data.maximum_size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice kd_tree_device = m_kd_tree_data.to_device();

		m_nisml_data.to_device(kd_tree_device);

		kd_tree_device.core.user_settings = render_data.kd_tree_device.core.user_settings;

		return kd_tree_device;
	}

	IlluminationAwareKDTreeCoreDataHost<DataContainer> m_kd_tree_data;
	IlluminationAwareKDTreeNISMLDataHost<DataContainer> m_nisml_data;

	GenericSoA<DataContainer, unsigned int> m_counter_download_buffer;
};

#endif
