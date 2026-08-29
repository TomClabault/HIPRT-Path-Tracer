/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_SPLATTING_SAMPLE_SOA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_SPLATTING_SAMPLE_SOA_HOST_H

#include "Device/includes/ReSTIR/PG/SplattingSampleSoADevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRPGSplattingSampleSoAHostInternal = GenericSoA<DataContainer,
														  float3_t,	 // Position
														  float3_t,	 // Normal
														  float3_t>; // Incident direction

enum ReSTIRPGSplattingSampleSoAHostBuffers
{
	RESTIR_PG_SPLATTING_SAMPLE_POSITION,
	RESTIR_PG_SPLATTING_SAMPLE_NORMAL,
	RESTIR_PG_SPLATTING_SAMPLE_INCIDENT_DIRECTION,
};

template <template <typename> typename DataContainer>
struct ReSTIRPGSplattingSampleSoAHost
{
	void resize(unsigned int width, unsigned int height, unsigned int nb_bounces)
	{
		m_splatting_samples.resize(width * height * nb_bounces);

		m_splatting_samples.template memset_buffer<ReSTIRPGSplattingSampleSoAHostBuffers::RESTIR_PG_SPLATTING_SAMPLE_POSITION>(make_float3(0.0f, 0.0f, 0.0f));
		m_splatting_samples.template memset_buffer<ReSTIRPGSplattingSampleSoAHostBuffers::RESTIR_PG_SPLATTING_SAMPLE_NORMAL>(make_float3(0.0f, 0.0f, 0.0f));
		m_splatting_samples.template memset_buffer<ReSTIRPGSplattingSampleSoAHostBuffers::RESTIR_PG_SPLATTING_SAMPLE_INCIDENT_DIRECTION>(
			make_float3(0.0f, 0.0f, 0.0f));
	}

	void free()
	{
		m_splatting_samples.free();
	}

	std::size_t get_byte_size() const
	{
		return m_splatting_samples.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return m_splatting_samples.maximum_size();
	}

	ReSTIRPGSplattingSampleSoADevice to_device()
	{
		ReSTIRPGSplattingSampleSoADevice soa_device;

		soa_device.position = m_splatting_samples.template get_buffer_data_ptr<ReSTIRPGSplattingSampleSoAHostBuffers::RESTIR_PG_SPLATTING_SAMPLE_POSITION>();
		soa_device.normal	= m_splatting_samples.template get_buffer_data_ptr<ReSTIRPGSplattingSampleSoAHostBuffers::RESTIR_PG_SPLATTING_SAMPLE_NORMAL>();
		soa_device.incident_direction =
			m_splatting_samples.template get_buffer_data_ptr<ReSTIRPGSplattingSampleSoAHostBuffers::RESTIR_PG_SPLATTING_SAMPLE_INCIDENT_DIRECTION>();

		return soa_device;
	}

	ReSTIRPGSplattingSampleSoAHostInternal<DataContainer> m_splatting_samples;
};

#endif // #ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PG_SPLATTING_SAMPLE_SOA_HOST_H
