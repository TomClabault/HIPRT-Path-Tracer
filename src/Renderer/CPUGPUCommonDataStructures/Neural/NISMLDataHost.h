/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NEURAL_NISML_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_NEURAL_NISML_DATA_HOST_H

#include "Device/includes/Neural/NISML.h"
#include "Device/includes/Neural/NISML/NISMLDevice.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericAtomicType.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

#include <algorithm>
#include <type_traits>
#include <vector>

template <template <typename> typename DataContainer>
struct NISMLDataHost
{
	static constexpr unsigned int NISML_TRAINING_BATCH_SIZE = 1000000;

	void resize(unsigned int new_training_record_capacity = NISML_TRAINING_BATCH_SIZE)
	{
		GenericSoAHelpers::resize<DataContainer>(m_training_records, new_training_record_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_training_record_count, 1);
	}

	void reset()
	{
		if (maximum_size() == 0)
			return;

		if constexpr (std::is_same_v<DataContainer<unsigned int>, std::vector<unsigned int>>)
			m_training_record_count[0].store(0);
		else
			m_training_record_count.memset_whole_buffer(0);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_training_records		= DataContainer<NISMLTrainingSample>();
		m_training_record_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		return true;
	}

	std::size_t get_byte_size() const
	{
		return GenericSoAHelpers::get_byte_size(m_training_records) + GenericSoAHelpers::get_byte_size(m_training_record_count);
	}

	std::size_t maximum_size() const
	{
		return m_training_records.size();
	}

	unsigned int get_training_record_capacity() const
	{
		return static_cast<unsigned int>(m_training_records.size());
	}

	unsigned int get_training_record_count() const
	{
		if constexpr (std::is_same_v<DataContainer<unsigned int>, std::vector<unsigned int>>)
			return m_training_record_count[0].load();
		else
			return m_training_record_count.download_data()[0];
	}

	unsigned int get_effective_training_record_count() const
	{
		return std::min(get_training_record_count(), get_training_record_capacity());
	}

	NISMLDevice to_device()
	{
		NISMLDevice device;

		device.training_records			= GenericSoAHelpers::get_buffer_data_ptr(m_training_records);
		device.training_record_count	= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_training_record_count);
		device.training_record_capacity = get_training_record_capacity();

		return device;
	}

	DataContainer<NISMLTrainingSample> m_training_records;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_training_record_count;
};

#endif
