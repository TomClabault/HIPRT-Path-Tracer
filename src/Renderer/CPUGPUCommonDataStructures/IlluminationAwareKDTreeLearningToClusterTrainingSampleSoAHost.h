/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_SOA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_SOA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"

#include "HostDeviceCommon/Packing.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHostInternal =
	GenericSoA<DataContainer, float3_t, float3_t, unsigned int, unsigned int, unsigned int, unsigned int>;

enum IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHostBuffers
{
	ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_POSITIONS,
	ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_SHADING_NORMALS,
	ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_MESH_IDS,
	ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_VALID_FOR_LIGHT_CLUSTERING,
	ILLUMINATION_AWARE_KD_TREE_REPLAYED_LIGHTCUT_INDICES,
	ILLUMINATION_AWARE_KD_TREE_REPLAYED_LIGHTCUT_SLOTS
};

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHost
{
	void resize(unsigned int new_training_sample_capacity)
	{
		m_training_samples.resize(new_training_sample_capacity);
	}

	void free()
	{
		m_training_samples.free();
	}

	std::size_t get_byte_size() const
	{
		return m_training_samples.get_byte_size();
	}

	unsigned int maximum_size() const
	{
		return m_training_samples.maximum_size();
	}

	void to_device(IlluminationAwareKDTreeLearningToClusterDevice& learning_to_cluster_device)
	{
		learning_to_cluster_device.training_samples_soa.positions =
			m_training_samples.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_POSITIONS>();
		learning_to_cluster_device.training_samples_soa.shading_normals =
			m_training_samples.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_SHADING_NORMALS>();
		learning_to_cluster_device.training_samples_soa.mesh_ids =
			m_training_samples.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_MESH_IDS>();
		learning_to_cluster_device.training_samples_soa.valid_for_lightcut =
			m_training_samples.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_TRAINING_SAMPLE_VALID_FOR_LIGHT_CLUSTERING>();
		learning_to_cluster_device.training_samples_soa.replayed_lightcut_indices =
			m_training_samples.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_REPLAYED_LIGHTCUT_INDICES>();
		learning_to_cluster_device.training_samples_soa.replayed_lightcut_slots =
			m_training_samples.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_REPLAYED_LIGHTCUT_SLOTS>();
	}

	IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHostInternal<DataContainer> m_training_samples;
};

#endif // #ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_TRAINING_SAMPLE_SOA_HOST_H
