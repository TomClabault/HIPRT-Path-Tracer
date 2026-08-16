/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_NISML_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_NISML_DATA_HOST_H

#include "Device/includes/HashGrid.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

#include <algorithm>
#include <type_traits>
#include <vector>

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeNISMLDataHost
{
	void resize(unsigned int, unsigned int new_representative_capacity, unsigned int new_hash_table_reserved_bytes, unsigned int new_hash_normal_precision)
	{
		m_representative_capacity	= new_representative_capacity > 0u ? new_representative_capacity : 1u;
		m_hash_table_reserved_bytes = new_hash_table_reserved_bytes;
		m_hash_normal_precision		= std::clamp(new_hash_normal_precision, 1u, 5u);

		// Representative cache payload for every representative stored in the hash entry.
		std::size_t bytes_per_hash_entry = sizeof(IlluminationAwareKDTreeNISMLCache) * m_representative_capacity;
		// Hash key, representative sample count, and representative occupied count.
		bytes_per_hash_entry += sizeof(GenericAtomicType<unsigned int, DataContainer>) * 3;
		// Representative dirty flag.
		bytes_per_hash_entry += sizeof(GenericAtomicType<unsigned char, DataContainer>);
		// Hash entry initialization state.
		bytes_per_hash_entry += sizeof(GenericAtomicType<unsigned int, DataContainer>);
		// Cache-ready flag.
		bytes_per_hash_entry += sizeof(unsigned char);
		// Per-representative valid flag and replacement write lock.
		bytes_per_hash_entry += (sizeof(unsigned char) + sizeof(GenericAtomicType<unsigned int, DataContainer>)) * m_representative_capacity;

		std::size_t hash_table_capacity	 = m_hash_table_reserved_bytes / bytes_per_hash_entry;
		m_hash_table_capacity			 = static_cast<unsigned int>(std::max<std::size_t>(hash_table_capacity, 1u));
		std::size_t representative_count = static_cast<std::size_t>(m_hash_table_capacity) * m_representative_capacity;

		GenericSoAHelpers::resize<DataContainer>(m_cache, representative_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_sample_counts, m_hash_table_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_occupied_counts, m_hash_table_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_valid, representative_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_write_locks, representative_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_dirty, m_hash_table_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_cache_ready, m_hash_table_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_hash_keys, m_hash_table_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_hash_entry_states, m_hash_table_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_hash_occupied_entry_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_pending_cell_count, 1);
	}

	void clear_representative_metadata()
	{
		if (maximum_size() == 0)
			return;

		if constexpr (std::is_same_v<DataContainer<unsigned int>, std::vector<unsigned int>>)
		{
			for (GenericAtomicType<unsigned int, DataContainer>& hash_key : m_hash_keys)
				hash_key.store(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
			for (GenericAtomicType<unsigned int, DataContainer>& entry_state : m_hash_entry_states)
				entry_state.store(ILLUMINATION_AWARE_KD_TREE_NISML_HASH_ENTRY_EMPTY);
			m_hash_occupied_entry_count[0].store(0u);
			for (GenericAtomicType<unsigned int, DataContainer>& sample_count : m_representative_sample_counts)
				sample_count.store(0u);
			for (GenericAtomicType<unsigned int, DataContainer>& occupied_count : m_representative_occupied_counts)
				occupied_count.store(0u);
			std::fill(m_representative_valid.begin(), m_representative_valid.end(), 0u);
			for (GenericAtomicType<unsigned int, DataContainer>& write_lock : m_representative_write_locks)
				write_lock.store(0u);
			for (GenericAtomicType<unsigned char, DataContainer>& dirty : m_representative_dirty)
				dirty.store(0u);
			std::fill(m_cache_ready.begin(), m_cache_ready.end(), 0u);
			m_pending_cell_count[0].store(0u);
		}
		else
		{
			m_hash_keys.memset_whole_buffer(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
			m_hash_entry_states.memset_whole_buffer(ILLUMINATION_AWARE_KD_TREE_NISML_HASH_ENTRY_EMPTY);
			m_hash_occupied_entry_count.memset_whole_buffer(0u);
			m_representative_sample_counts.memset_whole_buffer(0u);
			m_representative_occupied_counts.memset_whole_buffer(0u);
			m_representative_valid.memset_whole_buffer(0u);
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
		m_hash_keys						 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_hash_entry_states				 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_hash_occupied_entry_count		 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_sample_counts	 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_occupied_counts = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_valid			 = DataContainer<unsigned char>();
		m_representative_write_locks	 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_dirty			 = DataContainer<GenericAtomicType<unsigned char, DataContainer>>();
		m_cache_ready					 = DataContainer<unsigned char>();
		m_pending_cell_count			 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_capacity		 = 1;
		m_hash_table_capacity			 = 0;

		return true;
	}

	std::size_t maximum_size() const
	{
		return m_hash_table_capacity;
	}

	void to_device(IlluminationAwareKDTreeDevice& kd_tree_device)
	{
		kd_tree_device.nisml.nisml_cache						  = GenericSoAHelpers::get_buffer_data_ptr(m_cache);
		kd_tree_device.nisml.nisml_hash_keys					  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_hash_keys);
		kd_tree_device.nisml.nisml_hash_entry_states			  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_hash_entry_states);
		kd_tree_device.nisml.nisml_hash_table_capacity			  = m_hash_table_capacity;
		kd_tree_device.nisml.nisml_hash_normal_precision		  = m_hash_normal_precision;
		kd_tree_device.nisml.nisml_hash_occupied_entry_count	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_hash_occupied_entry_count);
		kd_tree_device.nisml.nisml_representative_capacity		  = m_representative_capacity;
		kd_tree_device.nisml.nisml_representative_sample_counts	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_sample_counts);
		kd_tree_device.nisml.nisml_representative_occupied_counts = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_occupied_counts);
		kd_tree_device.nisml.nisml_representative_valid			  = GenericSoAHelpers::get_buffer_data_ptr(m_representative_valid);
		kd_tree_device.nisml.nisml_representative_write_locks	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_write_locks);
		kd_tree_device.nisml.nisml_representative_dirty			  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_dirty);
		kd_tree_device.nisml.nisml_cache_ready					  = GenericSoAHelpers::get_buffer_data_ptr(m_cache_ready);
		kd_tree_device.nisml.nisml_pending_cell_count			  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_pending_cell_count);
	}

	DataContainer<IlluminationAwareKDTreeNISMLCache> m_cache;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_hash_keys;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_hash_entry_states;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_hash_occupied_entry_count;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_sample_counts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_occupied_counts;
	DataContainer<unsigned char> m_representative_valid;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_write_locks;
	DataContainer<GenericAtomicType<unsigned char, DataContainer>> m_representative_dirty;
	DataContainer<unsigned char> m_cache_ready;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_pending_cell_count;
	unsigned int m_representative_capacity	 = 1;
	unsigned int m_hash_table_capacity		 = 0;
	unsigned int m_hash_table_reserved_bytes = 0;
	unsigned int m_hash_normal_precision	 = 2;
};

#endif
