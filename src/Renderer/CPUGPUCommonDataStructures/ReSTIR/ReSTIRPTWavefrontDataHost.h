/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_WAVEFRONT_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_WAVEFRONT_DATA_HOST_H

#include "Device/includes/ReSTIR/PT/Wavefront/ReSTIRPTWavefrontDataDevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRPTWavefrontDataHostInternal = GenericSoA<DataContainer,
													 ColorRGB32F, // Unweighted throughput up to the reconnection vertex
													 ColorRGB32F, // Weighted throughput after the reconnection vertex
													 ColorRGB32F, // Pending throughput
													 float,		  // Reservoir weight sum
													 float3_t>;	  // Full precision ray direction

enum ReSTIRPTWavefrontDataHostBuffers
{
	RESTIR_PT_WAVEFRONT_PATH_UNWEIGHTED_THROUGHPUTS_UP_TO_RC_VERTEX,
	RESTIR_PT_WAVEFRONT_PATH_WEIGHTED_THROUGHPUTS_AFTER_RC_VERTEX,
	RESTIR_PT_WAVEFRONT_PATH_PENDING_THROUGHPUTS,
	RESTIR_PT_WAVEFRONT_PATH_RESERVOIR_WEIGHT_SUMS,
	RESTIR_PT_WAVEFRONT_PATH_RAY_DIRECTIONS,
};

template <template <typename> typename DataContainer>
struct ReSTIRPTWavefrontDataHost
{
	void resize(unsigned int path_capacity)
	{
		m_restir_pt_wavefront_data.resize(path_capacity);
	}

	bool free()
	{
		if (get_byte_size() == 0)
			return false;

		m_restir_pt_wavefront_data.free();

		return true;
	}

	std::size_t get_byte_size() const
	{
		return m_restir_pt_wavefront_data.get_byte_size();
	}

	std::size_t path_capacity() const
	{
		return m_restir_pt_wavefront_data.template get_buffer<RESTIR_PT_WAVEFRONT_PATH_RESERVOIR_WEIGHT_SUMS>().size();
	}

	ReSTIRPTWavefrontDataDevice to_device()
	{
		ReSTIRPTWavefrontDataDevice restir_pt_wavefront_data_device;
		if (path_capacity() == 0)
			return restir_pt_wavefront_data_device;

		restir_pt_wavefront_data_device.path_unweighted_throughputs_up_to_rc_vertex =
			m_restir_pt_wavefront_data.template get_buffer_data_ptr<RESTIR_PT_WAVEFRONT_PATH_UNWEIGHTED_THROUGHPUTS_UP_TO_RC_VERTEX>();
		restir_pt_wavefront_data_device.path_weighted_throughputs_after_rc_vertex =
			m_restir_pt_wavefront_data.template get_buffer_data_ptr<RESTIR_PT_WAVEFRONT_PATH_WEIGHTED_THROUGHPUTS_AFTER_RC_VERTEX>();
		restir_pt_wavefront_data_device.path_pending_throughputs =
			m_restir_pt_wavefront_data.template get_buffer_data_ptr<RESTIR_PT_WAVEFRONT_PATH_PENDING_THROUGHPUTS>();
		restir_pt_wavefront_data_device.path_reservoir_weight_sums =
			m_restir_pt_wavefront_data.template get_buffer_data_ptr<RESTIR_PT_WAVEFRONT_PATH_RESERVOIR_WEIGHT_SUMS>();
		restir_pt_wavefront_data_device.path_ray_directions =
			m_restir_pt_wavefront_data.template get_buffer_data_ptr<RESTIR_PT_WAVEFRONT_PATH_RAY_DIRECTIONS>();

		return restir_pt_wavefront_data_device;
	}

	ReSTIRPTWavefrontDataHostInternal<DataContainer> m_restir_pt_wavefront_data;
};

#endif // #ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_WAVEFRONT_DATA_HOST_H
