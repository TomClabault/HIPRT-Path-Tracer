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
	float3, // sum points
	GenericAtomicType<unsigned int, DataContainer>, // num points
	GenericAtomicType<int, DataContainer>,  // primitive index
	float3, // world points
	Octahedral24BitNormalPadded32b,  // world normals
	float,  // roughness
	float,  // metallic
	float,  // specular
	GenericAtomicType<unsigned int, DataContainer>, // hash keys
	GenericAtomicType<unsigned int, DataContainer>, // grid cells alive
	unsigned int // grid cells alive list
>;

enum ReGIRHashCellDataSoAHostBuffers
{
	REGIR_HASH_CELL_SUM_POINTS,
	REGIR_HASH_CELL_NUM_POINTS,

	REGIR_HASH_CELL_PRIM_INDEX,
	REGIR_HASH_CELL_POINTS,
	REGIR_HASH_CELL_NORMALS,
	REGIR_HASH_CELL_ROUGHNESS,
	REGIR_HASH_CELL_METALLIC,
	REGIR_HASH_CELL_SPECULAR,
	REGIR_HASH_CELL_HASH_KEYS,

	REGIR_HASH_CELLS_ALIVE,
	REGIR_HASH_CELLS_ALIVE_LIST
};

template <template <typename> typename DataContainer>
struct ReGIRHashCellDataSoAHost
{
	void resize(unsigned int new_number_of_cells)
	{
		new_number_of_cells = hippt::max(new_number_of_cells, 1u);

		m_hash_cell_data.resize(new_number_of_cells);

		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_HASH_KEYS>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_PRIM_INDEX>(ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE);
		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>(0u);
		m_hash_cell_data.template memset_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

		// resize and memset_buffer static call
		GenericSoAHelpers::resize<DataContainer>(m_grid_cells_alive_count, 1);
		GenericSoAHelpers::memset_buffer<DataContainer>(m_grid_cells_alive_count, 0u);
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

		hash_cell_data.sum_points = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_SUM_POINTS>();
		hash_cell_data.num_points = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_NUM_POINTS>();

		hash_cell_data.hit_primitive = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_PRIM_INDEX>();
		hash_cell_data.world_points = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_POINTS>();
		hash_cell_data.world_normals = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_NORMALS>();
		hash_cell_data.roughness = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_ROUGHNESS>();
		hash_cell_data.metallic = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_METALLIC>();
		hash_cell_data.specular = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_SPECULAR>();
		hash_cell_data.checksums = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_HASH_KEYS>();

		hash_cell_data.grid_cell_alive = m_hash_cell_data.template get_buffer_data_atomic_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>();
		hash_cell_data.grid_cells_alive_list = m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>();

		if constexpr (std::is_same_v<DataContainer<std::atomic<unsigned int>>, std::vector<std::atomic<unsigned int>>>)
			// This buffer is an std::vector so we can just call .data()
			hash_cell_data.grid_cells_alive_count = m_grid_cells_alive_count.data();
		else
			// For the GPU, we need to call .get_atomic_device_pointer()
			hash_cell_data.grid_cells_alive_count = m_grid_cells_alive_count.get_atomic_device_pointer();

		return hash_cell_data;
	}

	ReGIRHashCellDataSoAHostInternal<DataContainer> m_hash_cell_data;

	// Not in the SoA because this buffer's size doesn't follow the size of the other buffers.
	//
	// This one always just has size 1
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_grid_cells_alive_count;
};

#endif
