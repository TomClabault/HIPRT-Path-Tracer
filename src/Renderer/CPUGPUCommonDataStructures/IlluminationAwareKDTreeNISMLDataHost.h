/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_NISML_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_NISML_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeNISMLDataHost
{
	void resize(unsigned int new_node_capacity)
	{
		unsigned int cache_entry_count = new_node_capacity * ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT;

		GenericSoAHelpers::resize<DataContainer>(m_cache, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_sample_counts, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_write_locks, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_representative_ready, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_cache_ready, cache_entry_count);
		GenericSoAHelpers::resize<DataContainer>(m_pending_cell_count, 1);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_cache						   = DataContainer<IlluminationAwareKDTreeNISMLCache>();
		m_representative_sample_counts = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_write_locks   = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_ready		   = DataContainer<unsigned char>();
		m_cache_ready				   = DataContainer<unsigned char>();
		m_pending_cell_count		   = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		return true;
	}

	std::size_t maximum_size() const
	{
		return m_cache.size();
	}

	void to_device(IlluminationAwareKDTreeDevice& device)
	{
		device.nisml.nisml_cache						= GenericSoAHelpers::get_buffer_data_ptr(m_cache);
		device.nisml.nisml_representative_sample_counts = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_sample_counts);
		device.nisml.nisml_representative_write_locks	= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_write_locks);
		device.nisml.nisml_representative_ready			= GenericSoAHelpers::get_buffer_data_ptr(m_representative_ready);
		device.nisml.nisml_cache_ready					= GenericSoAHelpers::get_buffer_data_ptr(m_cache_ready);
		device.nisml.nisml_pending_cell_count			= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_pending_cell_count);
	}

	DataContainer<IlluminationAwareKDTreeNISMLCache> m_cache;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_sample_counts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_write_locks;
	DataContainer<unsigned char> m_representative_ready;
	DataContainer<unsigned char> m_cache_ready;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_pending_cell_count;
};

#endif
