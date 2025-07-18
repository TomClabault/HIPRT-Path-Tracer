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
// using ReGIRSampleSoAHost = GenericSoA<DataContainer, int, unsigned int>;
using ReGIRSampleSoAHost = GenericSoA<DataContainer, int, float3>;

template <template <typename> typename DataContainer>
using ReGIRReservoirSoAHost = GenericSoA<DataContainer, float>;

enum ReGIRSampleSoAHostBuffers
{
	REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX,
	// REGIR_SAMPLE_RANDOM_SEED
	REGIR_SAMPLE_POINT_ON_LIGHT
};

enum ReGIRReservoirSoAHostBuffers
{
	REGIR_RESERVOIR_UCW
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

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;
};

#endif
