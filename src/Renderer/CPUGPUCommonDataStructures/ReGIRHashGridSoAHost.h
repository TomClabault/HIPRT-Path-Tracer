/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_REGIR_HASH_GRID_SOA_CPU_GPU_H
#define RENDERER_REGIR_HASH_GRID_SOA_CPU_GPU_H

#include "Device/includes/ReSTIR/ReGIR/HashGridSoA.h"

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
	}

	void free()
	{
		samples.free();
		reservoirs.free();
	}

	std::size_t get_byte_size() const
	{
		return samples.get_byte_size() + reservoirs.get_byte_size();
	}

	unsigned int size_reservoirs() const
	{
		return samples.size();
	}

	ReGIRHashGridSoADevice to_device(float3 grid_resolution)
	{
		ReGIRHashGridSoADevice hash_grid_soa;

		hash_grid_soa.samples.emission = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSION>();
		hash_grid_soa.samples.emissive_triangle_index = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX>();
		hash_grid_soa.samples.light_area = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_LIGHT_AREA>();
		hash_grid_soa.samples.point_on_light = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_POINT_ON_LIGHT>();
		hash_grid_soa.samples.light_source_normal = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_LIGHT_SOURCE_NORMAL>();

		hash_grid_soa.reservoirs.UCW = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>();
		hash_grid_soa.reservoirs.M = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_M>();
		hash_grid_soa.reservoirs.number_of_reservoirs_per_cell = m_reservoirs_per_cell;

		hash_grid_soa.grid_resolution = grid_resolution;
		hash_grid_soa.m_total_number_of_cells = m_total_number_of_cells;

		return hash_grid_soa;
	}

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;

	unsigned int m_total_number_of_cells = 0;
	unsigned int m_reservoirs_per_cell = 0;
};

#endif
