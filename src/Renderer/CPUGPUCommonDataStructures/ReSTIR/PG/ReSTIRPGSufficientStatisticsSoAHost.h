/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_SSUFFICIENT_STATISTICS_SOA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_SSUFFICIENT_STATISTICS_SOA_HOST_H

#include "Device/includes/ReSTIR/PG/DistributionSufficientStatisticsSoADevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRPGSufficientStatisticsSoAHostInternal = GenericSoA<DataContainer,
															   GenericAtomicType<float, DataContainer>,			// directions_sums_x
															   GenericAtomicType<float, DataContainer>,			// directions_sums_y
															   GenericAtomicType<float, DataContainer>,			// directions_sums_z
															   GenericAtomicType<float, DataContainer>,			// responsibilities_sums
															   GenericAtomicType<unsigned int, DataContainer>>; // sample counts

enum ReSTIRPGSufficientStatisticsSoAHostBuffers
{
	RESTIR_PG_DIRECTIONS_SUMS_X,
	RESTIR_PG_DIRECTIONS_SUMS_Y,
	RESTIR_PG_DIRECTIONS_SUMS_Z,
	RESTIR_PG_RESPONSIBILITY_WEIGHTS_SUM,
	RESTIR_PG_SAMPLE_COUNT,
};

template <template <typename> typename DataContainer>
struct ReSTIRPGSufficientStatisticsSoAHost
{
	void resize(unsigned int new_number_of_cells, unsigned int distributions_component_count)
	{
		new_number_of_cells = hippt::max(new_number_of_cells, 1u);

		m_sufficient_statistics_data.resize(new_number_of_cells * distributions_component_count, { RESTIR_PG_SAMPLE_COUNT });

		m_sufficient_statistics_data.template memset_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_X>(0.0f);
		m_sufficient_statistics_data.template memset_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_Y>(0.0f);
		m_sufficient_statistics_data.template memset_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_Z>(0.0f);
		m_sufficient_statistics_data.template memset_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_RESPONSIBILITY_WEIGHTS_SUM>(0.0f);

		m_sufficient_statistics_data.template resize_one_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_SAMPLE_COUNT>(new_number_of_cells);
		m_sufficient_statistics_data.template memset_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_SAMPLE_COUNT>(0u);

		m_last_resize_component_count = distributions_component_count;
	}

	void free()
	{
		m_sufficient_statistics_data.free();
	}

	std::size_t get_byte_size() const
	{
		return m_sufficient_statistics_data.get_byte_size();
	}

	unsigned int size() const
	{
		return m_sufficient_statistics_data.size();
	}

	unsigned int get_last_resize_component_count() const
	{
		return m_last_resize_component_count;
	}

	ReSTIRPGDistributionSufficientStatisticsSoADevice to_device()
	{
		ReSTIRPGDistributionSufficientStatisticsSoADevice soa_device;

		soa_device.directions_sum_x =
			m_sufficient_statistics_data.template get_buffer_data_atomic_ptr<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_X>();
		soa_device.directions_sum_y =
			m_sufficient_statistics_data.template get_buffer_data_atomic_ptr<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_Y>();
		soa_device.directions_sum_z =
			m_sufficient_statistics_data.template get_buffer_data_atomic_ptr<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_Z>();
		soa_device.responsibility_weights_sum =
			m_sufficient_statistics_data
				.template get_buffer_data_atomic_ptr<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_RESPONSIBILITY_WEIGHTS_SUM>();
		soa_device.sample_count =
			m_sufficient_statistics_data.template get_buffer_data_atomic_ptr<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_SAMPLE_COUNT>();

		return soa_device;
	}

	ReSTIRPGSufficientStatisticsSoAHostInternal<DataContainer> m_sufficient_statistics_data;

	unsigned int m_last_resize_component_count = 0;
};

#endif
