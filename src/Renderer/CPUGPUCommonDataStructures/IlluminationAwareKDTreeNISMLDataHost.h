/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_NISML_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_NISML_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

#include <algorithm>
#include <type_traits>
#include <vector>

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeNISMLDataHost
{
	void resize(unsigned int new_node_capacity, unsigned int new_representative_capacity)
	{
		unsigned int cache_entry_count	 = new_node_capacity * ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT;
		m_representative_capacity		 = new_representative_capacity > 0u ? new_representative_capacity : 1u;
		std::size_t representative_count = static_cast<std::size_t>(cache_entry_count) * m_representative_capacity;

		GenericSoAHelpers::resize<DataContainer>(m_cache, representative_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_sample_counts, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_occupied_counts, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_write_locks, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_dirty, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_cache_ready, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_pending_cell_count, 1);
	}

	void clear_representative_metadata()
	{
		if (maximum_size() == 0)
			return;

		if constexpr (std::is_same_v<DataContainer<unsigned int>, std::vector<unsigned int>>)
		{
			for (GenericAtomicType<unsigned int, DataContainer>& sample_count : m_representative_sample_counts)
				sample_count.store(0u);
			for (unsigned int& occupied_count : m_representative_occupied_counts)
				occupied_count = 0u;
			for (GenericAtomicType<unsigned int, DataContainer>& write_lock : m_representative_write_locks)
				write_lock.store(0u);
			std::fill(m_representative_dirty.begin(), m_representative_dirty.end(), 0u);
			std::fill(m_cache_ready.begin(), m_cache_ready.end(), 0u);
			m_pending_cell_count[0].store(0u);
		}
		else
		{
			m_representative_sample_counts.memset_whole_buffer(0u);
			m_representative_occupied_counts.memset_whole_buffer(0u);
			m_representative_write_locks.memset_whole_buffer(0u);
			m_representative_dirty.memset_whole_buffer(0u);
			m_cache_ready.memset_whole_buffer(0u);
			m_pending_cell_count.memset_whole_buffer(0u);
		}
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_cache							 = DataContainer<IlluminationAwareKDTreeNISMLCache>();
		m_representative_sample_counts	 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_occupied_counts = DataContainer<unsigned int>();
		m_representative_write_locks	 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_dirty			 = DataContainer<unsigned char>();
		m_cache_ready					 = DataContainer<unsigned char>();
		m_pending_cell_count			 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_capacity		 = 1;

		return true;
	}

	std::size_t maximum_size() const
	{
		return m_cache.size();
	}

	void to_device(IlluminationAwareKDTreeDevice& kd_tree_device)
	{
		kd_tree_device.nisml.nisml_cache						  = GenericSoAHelpers::get_buffer_data_ptr(m_cache);
		kd_tree_device.nisml.nisml_representative_capacity		  = m_representative_capacity;
		kd_tree_device.nisml.nisml_representative_sample_counts	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_sample_counts);
		kd_tree_device.nisml.nisml_representative_occupied_counts = GenericSoAHelpers::get_buffer_data_ptr(m_representative_occupied_counts);
		kd_tree_device.nisml.nisml_representative_write_locks	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_write_locks);
		kd_tree_device.nisml.nisml_representative_dirty			  = GenericSoAHelpers::get_buffer_data_ptr(m_representative_dirty);
		kd_tree_device.nisml.nisml_cache_ready					  = GenericSoAHelpers::get_buffer_data_ptr(m_cache_ready);
		kd_tree_device.nisml.nisml_pending_cell_count			  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_pending_cell_count);
	}

	DataContainer<IlluminationAwareKDTreeNISMLCache> m_cache;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_sample_counts;
	DataContainer<unsigned int> m_representative_occupied_counts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_write_locks;
	DataContainer<unsigned char> m_representative_dirty;
	DataContainer<unsigned char> m_cache_ready;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_pending_cell_count;
	unsigned int m_representative_capacity = 1;
};

#endif
