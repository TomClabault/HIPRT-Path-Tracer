/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_WAVEFRONT_DATA_HOST_H
#define RENDERER_WAVEFRONT_DATA_HOST_H

#include "Device/includes/Wavefront/WavefrontDataDevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using WavefrontDataHostInternal = GenericSoA<DataContainer,
											 ColorRGB32F,
											 ColorRGB32F,
											 Octahedral24BitNormalPadded32b,
											 int,
											 float,
											 HitInfo,
											 unsigned int,
											 unsigned int,
											 unsigned char,
											 unsigned char,
											 unsigned int,
											 unsigned int,
											 GenericAtomicType<unsigned int, DataContainer>,
											 GenericAtomicType<unsigned int, DataContainer>>;

enum WavefrontDataHostBuffers
{
	WAVEFRONT_PATH_THROUGHPUTS,
	WAVEFRONT_PATH_RAY_COLORS,
	WAVEFRONT_PATH_RAY_DIRECTIONS,
	WAVEFRONT_PATH_BOUNCES,
	WAVEFRONT_PATH_ACCUMULATED_ROUGHNESSES,
	WAVEFRONT_PATH_CLOSEST_HIT_INFOS,
	WAVEFRONT_PATH_INTERSECTIONS_FOUND,
	WAVEFRONT_PATH_RNG_STATES,
	WAVEFRONT_PATH_VOLUME_STATE_BYTES,
	WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES,
	WAVEFRONT_PATH_QUEUE_0,
	WAVEFRONT_PATH_QUEUE_1,
	WAVEFRONT_QUEUE_COUNT_0,
	WAVEFRONT_QUEUE_COUNT_1,
};

template <template <typename> typename DataContainer>
struct WavefrontDataHost
{
	void resize(unsigned int path_capacity, std::size_t ray_volume_state_byte_size, std::size_t nee_deferred_mis_context_byte_size)
	{
		m_wavefront_data.resize(path_capacity, { WAVEFRONT_PATH_VOLUME_STATE_BYTES, WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES, WAVEFRONT_QUEUE_COUNT_0,
												 WAVEFRONT_QUEUE_COUNT_1 });
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_VOLUME_STATE_BYTES>(path_capacity * ray_volume_state_byte_size);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES>(path_capacity * nee_deferred_mis_context_byte_size);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_QUEUE_COUNT_0>(1);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_QUEUE_COUNT_1>(1);
	}

	bool free()
	{
		if (get_byte_size() == 0)
			return false;

		m_wavefront_data.free();
		return true;
	}

	std::size_t get_byte_size() const
	{
		return m_wavefront_data.get_byte_size();
	}

	std::size_t path_capacity() const
	{
		return m_wavefront_data.template get_buffer<WAVEFRONT_PATH_THROUGHPUTS>().size();
	}

	WavefrontDataDevice to_device()
	{
		WavefrontDataDevice wavefront_data_device;
		if (path_capacity() == 0)
			return wavefront_data_device;

		wavefront_data_device.path_throughputs			   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_THROUGHPUTS>();
		wavefront_data_device.path_ray_colors			   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RAY_COLORS>();
		wavefront_data_device.path_ray_directions		   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RAY_DIRECTIONS>();
		wavefront_data_device.path_bounces				   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_BOUNCES>();
		wavefront_data_device.path_accumulated_roughnesses = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_ACCUMULATED_ROUGHNESSES>();
		wavefront_data_device.path_closest_hit_infos	   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_CLOSEST_HIT_INFOS>();
		wavefront_data_device.path_intersections_found	   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_INTERSECTIONS_FOUND>();
		wavefront_data_device.path_rng_states			   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RNG_STATES>();
		wavefront_data_device.path_volume_states =
			reinterpret_cast<RayVolumeState*>(m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_VOLUME_STATE_BYTES>());
		wavefront_data_device.path_nee_deferred_mis_contexts = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES>();

		wavefront_data_device.path_queues[0]  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_QUEUE_0>();
		wavefront_data_device.path_queues[1]  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_QUEUE_1>();
		wavefront_data_device.queue_counts[0] = m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_QUEUE_COUNT_0>();
		wavefront_data_device.queue_counts[1] = m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_QUEUE_COUNT_1>();

		wavefront_data_device.path_capacity = static_cast<unsigned int>(path_capacity());
		return wavefront_data_device;
	}

	DataContainer<GenericAtomicType<unsigned int, DataContainer>>& get_queue_count_buffer(unsigned int queue_index)
	{
		if (queue_index == 0)
			return m_wavefront_data.template get_buffer<WAVEFRONT_QUEUE_COUNT_0>();

		return m_wavefront_data.template get_buffer<WAVEFRONT_QUEUE_COUNT_1>();
	}

	WavefrontDataHostInternal<DataContainer> m_wavefront_data;
};

#endif // #ifndef RENDERER_WAVEFRONT_DATA_HOST_H
