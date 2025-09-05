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

class ReGIRHashGridSoAHostUtils
{
public:
	static unsigned int get_bits_per_packed_emissive_triangle_global_index(unsigned int total_triangle_count)
	{
		return std::ceil(std::log2(total_triangle_count));
	}
};

template <template <typename> typename DataContainer>
struct ReGIRHashGridSoAHost
{
	void resize(unsigned int new_cell_count, unsigned int reservoirs_per_cell, unsigned int total_triangle_count)
	{
		new_cell_count = hippt::max(new_cell_count, 1u);

		m_total_number_of_cells = new_cell_count;
		m_reservoirs_per_cell = reservoirs_per_cell;
		m_total_triangle_count = total_triangle_count;

		samples.resize(m_total_number_of_cells * reservoirs_per_cell, { REGIR_SAMPLE_EMISSIVE_TRIANGLE_GLOBAL_INDEX });
		samples.template resize_one_buffer<REGIR_SAMPLE_EMISSIVE_TRIANGLE_GLOBAL_INDEX>(std::ceil(m_total_number_of_cells * reservoirs_per_cell * ReGIRHashGridSoAHostUtils::get_bits_per_packed_emissive_triangle_global_index(total_triangle_count) / (sizeof(ReGIRSampleSoADevice::ReGIRSampleEmissiveTriangleIndicesPackingType) * 8)));
		reservoirs.resize(m_total_number_of_cells * reservoirs_per_cell);

		// samples.template memset_buffer<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_GLOBAL_INDEX>(-1);
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
		out_soa_device.samples.emissive_triangle_indices_packed = samples.template get_buffer_data_atomic_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_GLOBAL_INDEX>();
		out_soa_device.samples.point_on_light = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_POINT_ON_LIGHT>();

		out_soa_device.samples.bits_per_emissive_triangle_global_index = ReGIRHashGridSoAHostUtils::get_bits_per_packed_emissive_triangle_global_index(m_total_triangle_count);

		out_soa_device.reservoirs.UCW = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>();
		out_soa_device.reservoirs.number_of_reservoirs_per_cell = m_reservoirs_per_cell;

		out_soa_device.m_total_number_of_cells = m_total_number_of_cells;
	}

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;

	unsigned int m_total_triangle_count = 0;
	unsigned int m_total_number_of_cells = 0;
	unsigned int m_reservoirs_per_cell = 0;
};

#endif
