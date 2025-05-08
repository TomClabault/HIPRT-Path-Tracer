/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_REGIR_HASH_CELL_DATA_SOA_CPU_GPU_H
#define RENDERER_REGIR_HASH_CELL_DATA_SOA_CPU_GPU_H

#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"

#include "HostDeviceCommon/Packing.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRGridBufferSoAHost.h"

template <template <typename> typename DataContainer>
using ReGIRHashCellDataSoAHostInternal = GenericSoA<DataContainer, 
	GenericAtomicType<float, DataContainer>,  // distance to center
	float3, // sum points
	GenericAtomicType<unsigned int, DataContainer>, // num points
	GenericAtomicType<int, DataContainer>,  // primitive
	float3, // world points
	Octahedral24BitNormalPadded32b,  // world normals
	GenericAtomicType<unsigned int, DataContainer> // hash keys
>;

enum ReGIRHashCellDataSoAHostBuffers
{
	REGIR_HASH_CELL_DISTANCE_TO_CENTER,
	REGIR_HASH_CELL_SUM_POINTS,
	REGIR_HASH_CELL_NUM_POINTS,

	REGIR_HASH_CELL_PRIM_INDEX,
	REGIR_HASH_CELL_POINTS,
	REGIR_HASH_CELL_NORMALS,
	REGIR_HASH_CELL_HASH_KEYS
};

template <template <typename> typename DataContainer>
struct ReGIRHashCellDataSoAHost
{
	void resize(unsigned int new_number_of_cells)
	{
		new_number_of_cells = hippt::max(new_number_of_cells, 1u);

		m_hash_cell_data.resize(new_number_of_cells);

		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_DISTANCE_TO_CENTER>(ReGIRHashCellDataSoADevice::UNDEFINED_DISTANCE);

		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_HASH_KEYS>(ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY);
		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_PRIM_INDEX>(ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE);
	}

	void free()
	{
		m_hash_cell_data.free();
	}

	std::size_t get_byte_size() const
	{
		return m_hash_cell_data.get_byte_size();
	}

	unsigned int size() const
	{
		return m_hash_cell_data.size();
	}

	ReGIRHashCellDataSoADevice to_device()
	{
		ReGIRHashCellDataSoADevice hash_cell_data;

		hash_cell_data.distance_to_center = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_DISTANCE_TO_CENTER>();
		hash_cell_data.sum_points = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_SUM_POINTS>();
		hash_cell_data.num_points = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_NUM_POINTS>();

		hash_cell_data.hit_primitive = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_PRIM_INDEX>();
		hash_cell_data.world_points = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_POINTS>();
		hash_cell_data.world_normals = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_NORMALS>();
		hash_cell_data.hash_keys = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_HASH_KEYS>();

		return hash_cell_data;
	}

	ReGIRHashCellDataSoAHostInternal<DataContainer> m_hash_cell_data;
};

#endif
