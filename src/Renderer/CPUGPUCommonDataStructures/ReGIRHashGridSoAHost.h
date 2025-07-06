/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_REGIR_HASH_GRID_SOA_CPU_GPU_H
#define RENDERER_REGIR_HASH_GRID_SOA_CPU_GPU_H

#include "Device/includes/ReSTIR/ReGIR/HashGridSoADevice.h"

#include "HostDeviceCommon/Packing.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRGridBufferSoAHost.h"

template <template <typename> typename DataContainer>
struct ReGIRHashGridSoAHost
{
	void resize(unsigned int new_cell_count, unsigned int reservoirs_per_cell)
	{
		new_cell_count = hippt::max(new_cell_count, 1u);

		m_total_number_of_cells = new_cell_count;
		m_reservoirs_per_cell = reservoirs_per_cell;

		samples.resize(m_total_number_of_cells * reservoirs_per_cell);
		reservoirs.resize(m_total_number_of_cells * reservoirs_per_cell);

		samples.template memset_buffer<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX>(-1);
		reservoirs.template memset_buffer<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>(ReGIRReservoir::UNDEFINED_UCW);
	}

	void free()
	{
		samples.free();
		reservoirs.free();

		m_total_number_of_cells = 0;
	}

	std::size_t get_byte_size() const
	{
		return samples.get_byte_size() + reservoirs.get_byte_size();
	}

	unsigned int size_reservoirs() const
	{
		return samples.size();
	}

	void to_device(ReGIRHashGridSoADevice& out_soa_device)
	{
		out_soa_device.samples.emissive_triangle_index = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX>();
		out_soa_device.samples.point_on_light = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_POINT_ON_LIGHT>();

		out_soa_device.reservoirs.UCW = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>();
		out_soa_device.reservoirs.number_of_reservoirs_per_cell = m_reservoirs_per_cell;

		out_soa_device.m_total_number_of_cells = m_total_number_of_cells;
	}

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;

	unsigned int m_total_number_of_cells = 0;
	unsigned int m_reservoirs_per_cell = 0;
};

#endif
