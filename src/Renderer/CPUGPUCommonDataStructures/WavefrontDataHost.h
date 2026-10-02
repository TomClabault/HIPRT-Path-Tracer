/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_WAVEFRONT_DATA_HOST_H
#define RENDERER_WAVEFRONT_DATA_HOST_H

#include "Device/includes/Wavefront/WavefrontDataDevice.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
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
											 unsigned int,
											 GenericAtomicType<unsigned int, DataContainer>,
											 GenericAtomicType<unsigned int, DataContainer>,
											 GenericAtomicType<unsigned int, DataContainer>,
											 unsigned int,
											 GenericAtomicType<unsigned int, DataContainer>,
											 float,
											 float,
											 float,
											 float,
											 float,
											 float,
											 unsigned int,
											 WavefrontResolvedMaterialClassification,
											 ColorRGB32F,
											 unsigned int,
											 unsigned int,
											 GenericAtomicType<unsigned int, DataContainer>,
											 WavefrontShadowWork,
											 GenericAtomicType<unsigned int, DataContainer>,
											 GenericAtomicType<unsigned int, DataContainer>,
											 WavefrontLearningToClusterShadowSidecar,
											 WavefrontNISMLShadowSidecar>;

enum WavefrontDataHostBuffers
{
	WAVEFRONT_PATH_THROUGHPUTS,
	WAVEFRONT_PATH_RAY_COLORS,
	WAVEFRONT_PATH_RAY_DIRECTIONS,
	WAVEFRONT_PATH_BOUNCES,
	WAVEFRONT_PATH_ACCUMULATED_ROUGHNESSES,
	WAVEFRONT_PATH_CLOSEST_HIT_INFOS,
	WAVEFRONT_PATH_STATE_FLAGS,
	WAVEFRONT_PATH_RNG_STATES,
	WAVEFRONT_PATH_VOLUME_STATE_BYTES,
	WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES,
	WAVEFRONT_PATH_QUEUE_0,
	WAVEFRONT_PATH_QUEUE_1,
	WAVEFRONT_PATH_COMPLETION_QUEUE,
	WAVEFRONT_QUEUE_COUNT_0,
	WAVEFRONT_QUEUE_COUNT_1,
	WAVEFRONT_COMPLETION_QUEUE_COUNT,
	WAVEFRONT_MATERIAL_FAMILY_INDICES,
	WAVEFRONT_MATERIAL_FAMILY_COUNTS,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_ROUGHNESS,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_METALLIC,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_COAT,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_SHEEN,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR_TRANSMISSION,
	WAVEFRONT_PATH_RESOLVED_MATERIAL_CONTROL_VALIDITY_MASKS,
	WAVEFRONT_PATH_CURRENT_MATERIAL_CLASSIFICATIONS,
	WAVEFRONT_PATH_SHADOW_THROUGHPUTS,
	WAVEFRONT_PATH_SHADOW_FLAGS,
	WAVEFRONT_PATH_PENDING_FINALIZE_RAY_STATES,
	WAVEFRONT_PATH_REMAINING_SHADOW_WORK_COUNTS,
	WAVEFRONT_SHADOW_WORK_ITEMS,
	WAVEFRONT_SHADOW_WORK_COUNT,
	WAVEFRONT_SHADOW_QUEUE_OVERFLOW_COUNT,
	WAVEFRONT_LEARNING_TO_CLUSTER_SHADOW_SIDECARS,
	WAVEFRONT_NISML_SHADOW_SIDECARS,
};

template <template <typename> typename DataContainer>
struct WavefrontDataHost
{
	void resize(unsigned int path_capacity,
				unsigned int shadow_work_capacity,
				std::size_t ray_volume_state_byte_size,
				std::size_t nee_deferred_mis_context_byte_size,
				bool allocate_resolved_material_control_cache,
				bool allocate_material_family_routing,
				bool allocate_learning_to_cluster_shadow_sidecars,
				bool allocate_nisml_shadow_sidecars)
	{
		m_wavefront_data.resize(path_capacity, { WAVEFRONT_PATH_VOLUME_STATE_BYTES,
												 WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES,
												 WAVEFRONT_QUEUE_COUNT_0,
												 WAVEFRONT_QUEUE_COUNT_1,
												 WAVEFRONT_COMPLETION_QUEUE_COUNT,
												 WAVEFRONT_MATERIAL_FAMILY_COUNTS,
												 WAVEFRONT_MATERIAL_FAMILY_INDICES,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_ROUGHNESS,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_METALLIC,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_COAT,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_SHEEN,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR_TRANSMISSION,
												 WAVEFRONT_PATH_RESOLVED_MATERIAL_CONTROL_VALIDITY_MASKS,
												 WAVEFRONT_PATH_CURRENT_MATERIAL_CLASSIFICATIONS,
												 WAVEFRONT_PATH_SHADOW_THROUGHPUTS,
												 WAVEFRONT_PATH_SHADOW_FLAGS,
												 WAVEFRONT_PATH_PENDING_FINALIZE_RAY_STATES,
												 WAVEFRONT_PATH_REMAINING_SHADOW_WORK_COUNTS,
												 WAVEFRONT_SHADOW_WORK_ITEMS,
												 WAVEFRONT_SHADOW_WORK_COUNT,
												 WAVEFRONT_SHADOW_QUEUE_OVERFLOW_COUNT,
												 WAVEFRONT_LEARNING_TO_CLUSTER_SHADOW_SIDECARS,
												 WAVEFRONT_NISML_SHADOW_SIDECARS });
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_VOLUME_STATE_BYTES>(path_capacity * ray_volume_state_byte_size);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES>(path_capacity * nee_deferred_mis_context_byte_size);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_QUEUE_COUNT_0>(1);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_QUEUE_COUNT_1>(1);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_COMPLETION_QUEUE_COUNT>(1);
		unsigned int shadow_path_capacity = shadow_work_capacity > 0 ? path_capacity : 0;
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_SHADOW_THROUGHPUTS>(shadow_path_capacity);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_SHADOW_FLAGS>(shadow_path_capacity);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_PENDING_FINALIZE_RAY_STATES>(shadow_path_capacity);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_REMAINING_SHADOW_WORK_COUNTS>(shadow_path_capacity);
		m_wavefront_data.template memset_buffer<WAVEFRONT_PATH_REMAINING_SHADOW_WORK_COUNTS>(0u);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_SHADOW_WORK_ITEMS>(shadow_work_capacity);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_SHADOW_WORK_COUNT>(shadow_work_capacity > 0 ? 1 : 0);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_SHADOW_QUEUE_OVERFLOW_COUNT>(shadow_work_capacity > 0 ? 1 : 0);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_LEARNING_TO_CLUSTER_SHADOW_SIDECARS>(
			allocate_learning_to_cluster_shadow_sidecars ? shadow_work_capacity : 0);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_NISML_SHADOW_SIDECARS>(allocate_nisml_shadow_sidecars ? shadow_work_capacity : 0);
		std::size_t family_path_count = allocate_material_family_routing ? static_cast<std::size_t>(path_capacity) * KernelMaterialSpecializationCount : 0;
		std::size_t family_count	  = allocate_material_family_routing ? KernelMaterialSpecializationCount : 0;
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_MATERIAL_FAMILY_INDICES>(family_path_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_MATERIAL_FAMILY_COUNTS>(family_count);
		std::size_t resolved_control_count = allocate_resolved_material_control_cache ? path_capacity : 0;
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_ROUGHNESS>(resolved_control_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_METALLIC>(resolved_control_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR>(resolved_control_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_COAT>(resolved_control_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_SHEEN>(resolved_control_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR_TRANSMISSION>(resolved_control_count);
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_RESOLVED_MATERIAL_CONTROL_VALIDITY_MASKS>(resolved_control_count);
		std::size_t current_material_classification_count = allocate_resolved_material_control_cache && allocate_material_family_routing ? path_capacity : 0;
		m_wavefront_data.template resize_one_buffer<WAVEFRONT_PATH_CURRENT_MATERIAL_CLASSIFICATIONS>(current_material_classification_count);
		m_resolved_material_control_cache_allocated		= allocate_resolved_material_control_cache;
		m_material_family_routing_allocated				= allocate_material_family_routing;
		m_shadow_work_capacity							= shadow_work_capacity;
		m_learning_to_cluster_shadow_sidecars_allocated = allocate_learning_to_cluster_shadow_sidecars;
		m_nisml_shadow_sidecars_allocated				= allocate_nisml_shadow_sidecars;
	}

	bool free()
	{
		if (get_byte_size() == 0)
			return false;

		m_wavefront_data.free();
		m_resolved_material_control_cache_allocated		= false;
		m_material_family_routing_allocated				= false;
		m_shadow_work_capacity							= 0;
		m_learning_to_cluster_shadow_sidecars_allocated = false;
		m_nisml_shadow_sidecars_allocated				= false;
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

	bool has_resolved_material_control_cache() const
	{
		return m_resolved_material_control_cache_allocated;
	}

	bool has_material_family_routing() const
	{
		return m_material_family_routing_allocated;
	}

	unsigned int shadow_work_capacity() const
	{
		return m_shadow_work_capacity;
	}

	bool has_learning_to_cluster_shadow_sidecars() const
	{
		return m_learning_to_cluster_shadow_sidecars_allocated;
	}

	bool has_nisml_shadow_sidecars() const
	{
		return m_nisml_shadow_sidecars_allocated;
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
		wavefront_data_device.path_state_flags			   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_STATE_FLAGS>();
		wavefront_data_device.path_rng_states			   = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RNG_STATES>();
		wavefront_data_device.path_volume_states =
			reinterpret_cast<RayVolumeState*>(m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_VOLUME_STATE_BYTES>());
		wavefront_data_device.path_nee_deferred_mis_contexts = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_NEE_DEFERRED_MIS_CONTEXT_BYTES>();
		if (m_material_family_routing_allocated)
		{
			wavefront_data_device.material_family_indices		  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_MATERIAL_FAMILY_INDICES>();
			wavefront_data_device.material_family_counts		  = m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_MATERIAL_FAMILY_COUNTS>();
			wavefront_data_device.material_family_routing_enabled = 1;
		}
		if (m_resolved_material_control_cache_allocated)
		{
			wavefront_data_device.path_resolved_material_roughness =
				m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_ROUGHNESS>();
			wavefront_data_device.path_resolved_material_metallic = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_METALLIC>();
			wavefront_data_device.path_resolved_material_specular = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR>();
			wavefront_data_device.path_resolved_material_coat	  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_COAT>();
			wavefront_data_device.path_resolved_material_sheen	  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_SHEEN>();
			wavefront_data_device.path_resolved_material_specular_transmission =
				m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_SPECULAR_TRANSMISSION>();
			wavefront_data_device.path_resolved_material_control_validity_masks =
				m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_RESOLVED_MATERIAL_CONTROL_VALIDITY_MASKS>();
		}
		if (m_resolved_material_control_cache_allocated && m_material_family_routing_allocated)
			wavefront_data_device.path_current_material_classifications =
				m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_CURRENT_MATERIAL_CLASSIFICATIONS>();

		wavefront_data_device.path_queues[0]								= m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_QUEUE_0>();
		wavefront_data_device.path_queues[1]								= m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_QUEUE_1>();
		wavefront_data_device.path_queues[WAVEFRONT_COMPLETION_QUEUE_INDEX] = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_COMPLETION_QUEUE>();
		wavefront_data_device.queue_counts[0]								= m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_QUEUE_COUNT_0>();
		wavefront_data_device.queue_counts[1]								= m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_QUEUE_COUNT_1>();
		wavefront_data_device.queue_counts[WAVEFRONT_COMPLETION_QUEUE_INDEX] =
			m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_COMPLETION_QUEUE_COUNT>();

		if (m_shadow_work_capacity > 0)
		{
			wavefront_data_device.path_shadow_throughputs = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_SHADOW_THROUGHPUTS>();
			wavefront_data_device.path_shadow_flags		  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_SHADOW_FLAGS>();
			wavefront_data_device.path_pending_finalize_ray_states =
				m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_PATH_PENDING_FINALIZE_RAY_STATES>();
			wavefront_data_device.path_remaining_shadow_work_counts =
				m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_PATH_REMAINING_SHADOW_WORK_COUNTS>();

			wavefront_data_device.shadow_work_items			  = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_SHADOW_WORK_ITEMS>();
			wavefront_data_device.shadow_work_count			  = m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_SHADOW_WORK_COUNT>();
			wavefront_data_device.shadow_queue_overflow_count = m_wavefront_data.template get_buffer_data_atomic_ptr<WAVEFRONT_SHADOW_QUEUE_OVERFLOW_COUNT>();
			wavefront_data_device.shadow_work_capacity		  = m_shadow_work_capacity;

			if (m_learning_to_cluster_shadow_sidecars_allocated)
				wavefront_data_device.learning_to_cluster_shadow_sidecars =
					m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_LEARNING_TO_CLUSTER_SHADOW_SIDECARS>();
			if (m_nisml_shadow_sidecars_allocated)
				wavefront_data_device.nisml_shadow_sidecars = m_wavefront_data.template get_buffer_data_ptr<WAVEFRONT_NISML_SHADOW_SIDECARS>();
		}

		wavefront_data_device.path_capacity = static_cast<unsigned int>(path_capacity());
		return wavefront_data_device;
	}

	DataContainer<GenericAtomicType<unsigned int, DataContainer>>& get_queue_count_buffer(unsigned int queue_index)
	{
		if (queue_index == 0)
			return m_wavefront_data.template get_buffer<WAVEFRONT_QUEUE_COUNT_0>();
		if (queue_index == WAVEFRONT_COMPLETION_QUEUE_INDEX)
			return m_wavefront_data.template get_buffer<WAVEFRONT_COMPLETION_QUEUE_COUNT>();

		return m_wavefront_data.template get_buffer<WAVEFRONT_QUEUE_COUNT_1>();
	}

	DataContainer<GenericAtomicType<unsigned int, DataContainer>>& get_shadow_work_count_buffer()
	{
		return m_wavefront_data.template get_buffer<WAVEFRONT_SHADOW_WORK_COUNT>();
	}

	DataContainer<GenericAtomicType<unsigned int, DataContainer>>& get_shadow_queue_overflow_count_buffer()
	{
		return m_wavefront_data.template get_buffer<WAVEFRONT_SHADOW_QUEUE_OVERFLOW_COUNT>();
	}

	WavefrontDataHostInternal<DataContainer> m_wavefront_data;
	bool m_resolved_material_control_cache_allocated	 = false;
	bool m_material_family_routing_allocated			 = false;
	unsigned int m_shadow_work_capacity					 = 0;
	bool m_learning_to_cluster_shadow_sidecars_allocated = false;
	bool m_nisml_shadow_sidecars_allocated				 = false;
};

#endif // #ifndef RENDERER_WAVEFRONT_DATA_HOST_H
