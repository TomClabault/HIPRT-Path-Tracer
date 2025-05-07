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
using ReGIRHashCellDataSoAHost = GenericSoA<DataContainer, GenericAtomicType<int, DataContainer>, float3, Octahedral24BitNormalPadded32b, unsigned int>;

enum ReGIRRepresentativeSoAHostBuffers
{
	REGIR_HASH_CELL_PRIM_INDEX,
	REGIR_HASH_CELL_POINTS,
	REGIR_HASH_CELL_NORMALS,
	REGIR_HASH_CELL_HASH_KEYS
};

template <template <typename> typename DataContainer>
struct ReGIRHashGridSoAHost
{
	void resize(ReGIRSettings& settings, int overallocation_factor = 1)
	{
		m_total_number_of_cells = settings.grid_fill_grid.hash_grid.grid_resolution.x * settings.grid_fill_grid.hash_grid.grid_resolution.y * settings.grid_fill_grid.hash_grid.grid_resolution.z;
		m_total_number_of_cells *= overallocation_factor;

		samples.resize(m_total_number_of_cells * settings.get_number_of_reservoirs_per_cell());
		reservoirs.resize(m_total_number_of_cells * settings.get_number_of_reservoirs_per_cell());
		hash_cell_data.resize(m_total_number_of_cells);

		m_reservoirs_per_cell = settings.get_number_of_reservoirs_per_cell();
	}

	void free()
	{
		samples.free();
		reservoirs.free();
		hash_cell_data.free();
	}

	std::size_t get_byte_size() const
	{
		return samples.get_byte_size() + reservoirs.get_byte_size() + hash_cell_data.get_byte_size();
	}

	unsigned int size_reservoirs() const
	{
		return samples.size();
	}

	unsigned int size_cells() const
	{
		return hash_cell_data.size();
	}

	unsigned int get_total_number_of_cells(ReGIRSettings& settings, int overallocation_factor = 1) const
	{
		return settings.grid_fill_grid.hash_grid.grid_resolution.x * settings.grid_fill_grid.hash_grid.grid_resolution.y * settings.grid_fill_grid.hash_grid.grid_resolution.z * overallocation_factor;
	}

	ReGIRHashGridSoADevice to_device(ReGIRSettings& regir_settings)
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

		regir_settings.hash_cell_data.representative_primitive = hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_HASH_CELL_PRIM_INDEX>();
		regir_settings.hash_cell_data.representative_points = hash_cell_data.template get_buffer_data_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_HASH_CELL_POINTS>();
		regir_settings.hash_cell_data.representative_normals = hash_cell_data.template get_buffer_data_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_HASH_CELL_NORMALS>();
		regir_settings.hash_cell_data.hash_keys = hash_cell_data.template get_buffer_data_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_HASH_CELL_HASH_KEYS>();

		hash_grid_soa.m_total_number_of_cells = m_total_number_of_cells;

		return hash_grid_soa;
	}

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;
	ReGIRHashCellDataSoAHost<DataContainer> hash_cell_data;

	unsigned int m_total_number_of_cells = 0;
	unsigned int m_reservoirs_per_cell = 0;
};

#endif
