/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_REGIR_PRESAMPLED_LIGHTS_SOA_CPU_GPU_H
#define RENDERER_REGIR_PRESAMPLED_LIGHTS_SOA_CPU_GPU_H

#include "HostDeviceCommon/Packing.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReGIRPresampledLightsSoAHostInternal = GenericSoA<DataContainer,
	int,  // primitive index
	float, // light area
	float3, // point on light
	Octahedral24BitNormalPadded32b // light normals
>;

enum ReGIRPresampledLightsSoAHostBuffers
{
	REGIR_PRESAMPLED_LIGHTS_TRIANGLE_INDEX,
	REGIR_PRESAMPLED_LIGHTS_LIGHT_AREA,
	REGIR_PRESAMPLED_LIGHTS_POINT_ON_LIGHT,
	REGIR_PRESAMPLED_LIGHTS_LIGHT_NORMAL,
};

template <template <typename> typename DataContainer>
struct ReGIRPresampledLightsSoAHost
{
	void resize(unsigned int new_presampled_lights_count)
	{
		m_hash_cell_data.resize(new_presampled_lights_count);
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

	void to_device(ReGIRPresampledLightsSoADevice& soa_device)
	{
		soa_device.emissive_triangle_index = m_hash_cell_data.template get_buffer_data_ptr<ReGIRPresampledLightsSoAHostBuffers::REGIR_PRESAMPLED_LIGHTS_TRIANGLE_INDEX>();
		soa_device.light_area = m_hash_cell_data.template get_buffer_data_ptr<ReGIRPresampledLightsSoAHostBuffers::REGIR_PRESAMPLED_LIGHTS_LIGHT_AREA>();
		soa_device.point_on_light = m_hash_cell_data.template get_buffer_data_ptr<ReGIRPresampledLightsSoAHostBuffers::REGIR_PRESAMPLED_LIGHTS_POINT_ON_LIGHT>();
		soa_device.light_normal = m_hash_cell_data.template get_buffer_data_ptr<ReGIRPresampledLightsSoAHostBuffers::REGIR_PRESAMPLED_LIGHTS_LIGHT_NORMAL>();
	}

	ReGIRPresampledLightsSoAHostInternal<DataContainer> m_hash_cell_data;
};

#endif
