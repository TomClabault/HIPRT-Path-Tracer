/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_REGIR_RESERVOIR_SOA_CPU_GPU_H
#define RENDERER_REGIR_RESERVOIR_SOA_CPU_GPU_H

#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"
#include "HostDeviceCommon/Packing.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReGIRSampleSoAHost = GenericSoA<DataContainer, Float3xLengthUint10bPacked, int, float, float3, Octahedral24BitNormalPadded32b>;

template <template <typename> typename DataContainer>
using ReGIRReservoirSoAHost = GenericSoA<DataContainer, float, unsigned char>;

enum ReGIRSampleSoAHostBuffers
{
	REGIR_SAMPLE_EMISSION,
	REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX,
	REGIR_SAMPLE_LIGHT_AREA,
	REGIR_SAMPLE_POINT_ON_LIGHT,
	REGIR_SAMPLE_LIGHT_SOURCE_NORMAL
};

enum ReGIRReservoirSoAHostBuffers
{
	REGIR_RESERVOIR_UCW,
	REGIR_RESERVOIR_M
};

template <template <typename> typename DataContainer>
struct ReGIRGridBufferSoAHost
{
	void resize(int new_element_count)
	{
		samples.resize(new_element_count);
		reservoirs.resize(new_element_count);
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

	unsigned int size() const
	{
		return samples.size();
	}

	// ReGIRGridBufferSoADevice to_device()
	// {
	// 	ReGIRGridBufferSoADevice grid_buffer_soa;

	// 	grid_buffer_soa.samples.emission = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSION>().data();
	// 	grid_buffer_soa.samples.emissive_triangle_index = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX>().data();
	// 	grid_buffer_soa.samples.light_area = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_LIGHT_AREA>().data();
	// 	grid_buffer_soa.samples.point_on_light = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_POINT_ON_LIGHT>().data();
	// 	grid_buffer_soa.samples.light_source_normal = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_LIGHT_SOURCE_NORMAL>().data();

	// 	grid_buffer_soa.reservoirs.UCW = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>().data();
	// 	grid_buffer_soa.reservoirs.M = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_M>().data();

	// 	return grid_buffer_soa;
	// }

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;
};

#endif
