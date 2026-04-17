/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_DISTRIBUTION_SOA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_DISTRIBUTION_SOA_HOST_H

#include "Device/includes/ReSTIR/PG/DistributionSoADevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRPGDistributionSoAHostInternal = GenericSoA<DataContainer,
													   float3_t, // VMF axis
													   float,	 // VMF sharpness
													   float>;	 // VMF component weight

enum ReSTIRPGDistributionSoAHostBuffers
{
	RESTIR_PG_DISTRIBUTION_VMF_AXIS,
	RESTIR_PG_DISTRIBUTION_VMF_SHARPNESS,
	RESTIR_PG_DISTRIBUTION_VMF_COMPONENT_WEIGHT,
};

template <template <typename> typename DataContainer>
struct ReSTIRPGDistributionSoAHost
{
	void resize(unsigned int new_number_of_cells, unsigned int distributions_component_count)
	{
		new_number_of_cells = hippt::max(new_number_of_cells, 1u);

		m_distribution_data.resize(new_number_of_cells * distributions_component_count);

		m_distribution_data.template memset_buffer<ReSTIRPGDistributionSoAHostBuffers::RESTIR_PG_DISTRIBUTION_VMF_AXIS>(make_float3(0.0f, 0.0f, 0.0f));
		m_distribution_data.template memset_buffer<ReSTIRPGDistributionSoAHostBuffers::RESTIR_PG_DISTRIBUTION_VMF_SHARPNESS>(0.0f);
		m_distribution_data.template memset_buffer<ReSTIRPGDistributionSoAHostBuffers::RESTIR_PG_DISTRIBUTION_VMF_COMPONENT_WEIGHT>(0.0f);

		m_last_resize_component_count = distributions_component_count;
		m_last_resize_number_of_cells = distributions_component_count;
	}

	void free()
	{
		m_distribution_data.free();
		m_last_resize_component_count = 0;
	}

	std::size_t get_byte_size() const
	{
		return m_distribution_data.get_byte_size();
	}

	unsigned int get_total_element_count() const
	{
		return get_last_resize_component_count() * get_last_resize_number_of_cells();
	}

	unsigned int get_last_resize_component_count() const
	{
		return m_last_resize_component_count;
	}

	unsigned int get_last_resize_number_of_cells() const
	{
		return m_last_resize_number_of_cells;
	}

	ReSTIRPGDistributionSoADevice to_device()
	{
		ReSTIRPGDistributionSoADevice soa_device;

		soa_device.axis		 = m_distribution_data.template get_buffer_data_ptr<ReSTIRPGDistributionSoAHostBuffers::RESTIR_PG_DISTRIBUTION_VMF_AXIS>();
		soa_device.sharpness = m_distribution_data.template get_buffer_data_ptr<ReSTIRPGDistributionSoAHostBuffers::RESTIR_PG_DISTRIBUTION_VMF_SHARPNESS>();
		soa_device.component_weight =
			m_distribution_data.template get_buffer_data_ptr<ReSTIRPGDistributionSoAHostBuffers::RESTIR_PG_DISTRIBUTION_VMF_COMPONENT_WEIGHT>();

		return soa_device;
	}

	ReSTIRPGDistributionSoAHostInternal<DataContainer> m_distribution_data;

	unsigned int m_last_resize_component_count = 0;
	unsigned int m_last_resize_number_of_cells = 0;
};

#endif
