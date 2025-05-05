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
using ReGIRRepresentativeSoAHost = GenericSoA<DataContainer, GenericAtomicType<float, DataContainer>, GenericAtomicType<int, DataContainer>, unsigned int, Octahedral24BitNormalPadded32b>;

enum ReGIRRepresentativeSoAHostBuffers
{
	REGIR_REPRESENTATIVE_DISTANCE_TO_CENTER,
	REGIR_REPRESENTATIVE_PRIM_INDEX,
	REGIR_REPRESENTATIVE_POINTS,
	REGIR_REPRESENTATIVE_NORMALS
};

template <template <typename> typename DataContainer>
struct ReGIRHashGridSoAHost
{
	void resize(int new_number_of_cells, int num_reservoirs_per_cell)
	{
		samples.resize(new_number_of_cells * num_reservoirs_per_cell);
		reservoirs.resize(new_number_of_cells * num_reservoirs_per_cell);
		representative.resize(new_number_of_cells);
	}

	void free()
	{
		samples.free();
		reservoirs.free();
		representative.free();
	}

	std::size_t get_byte_size() const
	{
		return samples.get_byte_size() + reservoirs.get_byte_size() + representative.get_byte_size();
	}

	unsigned int size_reservoirs() const
	{
		return samples.size();
	}

	unsigned int size_cells() const
	{
		return representative.size();
	}

	ReGIRHashGridSoADevice to_device()
	{
		ReGIRHashGridSoADevice hash_grid_soa;

		hash_grid_soa.samples.emission = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSION>();
		hash_grid_soa.samples.emissive_triangle_index = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_EMISSIVE_TRIANGLE_INDEX>();
		hash_grid_soa.samples.light_area = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_LIGHT_AREA>();
		hash_grid_soa.samples.point_on_light = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_POINT_ON_LIGHT>();
		hash_grid_soa.samples.light_source_normal = samples.template get_buffer_data_ptr<ReGIRSampleSoAHostBuffers::REGIR_SAMPLE_LIGHT_SOURCE_NORMAL>();

		hash_grid_soa.reservoirs.UCW = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>();
		hash_grid_soa.reservoirs.M = reservoirs.template get_buffer_data_ptr<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_M>();

		hash_grid_soa.representative.distance_to_center = representative.template get_buffer_data_atomic_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_REPRESENTATIVE_DISTANCE_TO_CENTER>();
		hash_grid_soa.representative.representative_primitive = representative.template get_buffer_data_atomic_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_REPRESENTATIVE_PRIM_INDEX>();
		hash_grid_soa.representative.representative_points = representative.template get_buffer_data_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_REPRESENTATIVE_POINTS>();
		hash_grid_soa.representative.representative_normals = representative.template get_buffer_data_ptr<ReGIRRepresentativeSoAHostBuffers::REGIR_REPRESENTATIVE_NORMALS>();

		return hash_grid_soa;
	}

	ReGIRSampleSoAHost<DataContainer> samples;
	ReGIRReservoirSoAHost<DataContainer> reservoirs;
	ReGIRRepresentativeSoAHost<DataContainer> representative;
};

#endif
