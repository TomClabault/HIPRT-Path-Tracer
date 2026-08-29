/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHost.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeLearningToClusterDataHost
{
	void resize(unsigned int new_node_capacity,
				unsigned int new_training_sample_capacity,
				unsigned int maximum_light_cut_size = LearningToClusterMaximumLightCutSize)
	{
		m_maximum_light_cut_size = maximum_light_cut_size;

		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_normal_clustering_set_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_samples, new_training_sample_capacity);
		m_learning_to_cluster_training_samples_soa.resize(new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_sample_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_initial_light_cut_node_indices, maximum_light_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_normal_clustering_sets, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_normal_face_observation_counts, static_cast<std::size_t>(new_node_capacity) * SurfaceNormalFace_Count);

		std::size_t light_clustering_capacity	= static_cast<std::size_t>(new_node_capacity) * 2;
		std::size_t light_cluster_slot_capacity = light_clustering_capacity * maximum_light_cut_size;

		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_node_indices, light_cluster_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_statistics, light_cluster_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_cdfs, light_cluster_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_data, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_sample_counts, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_shading_contexts, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_shading_context_states, light_clustering_capacity);
	}

	bool free()
	{
		bool light_clustering_data_freed = m_light_clustering_count.size() > 0;

		m_light_clustering_count							= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_normal_clustering_set_count						= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_learning_to_cluster_training_samples				= DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample>();
		m_learning_to_cluster_training_sample_count			= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		bool learning_to_cluster_training_samples_soa_freed = m_learning_to_cluster_training_samples_soa.maximum_size() > 0;
		m_learning_to_cluster_training_samples_soa.free();
		m_initial_light_cut_node_indices		= DataContainer<unsigned int>();
		m_normal_clustering_sets				= DataContainer<IlluminationAwareKDTreeNormalClusteringSet>();
		m_normal_face_observation_counts		= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_light_cluster_node_indices			= DataContainer<unsigned int>();
		m_light_cluster_statistics				= DataContainer<IlluminationAwareKDTreeLightClusterStatistics>();
		m_light_cluster_cdfs					= DataContainer<unsigned short int>();
		m_light_clustering_data					= DataContainer<IlluminationAwareKDTreeLightClusteringData>();
		m_light_cluster_sample_counts			= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_shading_contexts		= DataContainer<IlluminationAwareKDTreeSGShadingContext>();
		m_representative_shading_context_states = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		return light_clustering_data_freed || learning_to_cluster_training_samples_soa_freed;
	}

	std::size_t maximum_size() const
	{
		return m_light_clustering_data.size();
	}

	void to_device(IlluminationAwareKDTreeDevice& kd_tree_device)
	{
		kd_tree_device.learning_to_cluster.light_clustering_count		   = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_light_clustering_count);
		kd_tree_device.learning_to_cluster.light_clustering_capacity	   = static_cast<unsigned int>(m_light_clustering_data.size());
		kd_tree_device.learning_to_cluster.normal_clustering_sets		   = GenericSoAHelpers::get_buffer_data_ptr(m_normal_clustering_sets);
		kd_tree_device.learning_to_cluster.normal_clustering_set_count	   = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_normal_clustering_set_count);
		kd_tree_device.learning_to_cluster.normal_clustering_set_capacity  = static_cast<unsigned int>(m_normal_clustering_sets.size());
		kd_tree_device.learning_to_cluster.normal_face_observation_counts  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_normal_face_observation_counts);
		kd_tree_device.learning_to_cluster.initial_light_cut_node_indices  = GenericSoAHelpers::get_buffer_data_ptr(m_initial_light_cut_node_indices);
		kd_tree_device.learning_to_cluster.light_cluster_node_indices	   = GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_node_indices);
		kd_tree_device.learning_to_cluster.light_cluster_statistics		   = GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_statistics);
		kd_tree_device.learning_to_cluster.light_cluster_cdfs			   = GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_cdfs);
		kd_tree_device.learning_to_cluster.light_clustering_data		   = GenericSoAHelpers::get_buffer_data_ptr(m_light_clustering_data);
		kd_tree_device.learning_to_cluster.light_cluster_sample_counts	   = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_light_cluster_sample_counts);
		kd_tree_device.learning_to_cluster.representative_shading_contexts = GenericSoAHelpers::get_buffer_data_ptr(m_representative_shading_contexts);
		kd_tree_device.learning_to_cluster.representative_shading_context_states =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_shading_context_states);

		kd_tree_device.learning_to_cluster.training_samples = GenericSoAHelpers::get_buffer_data_ptr(m_learning_to_cluster_training_samples);
		m_learning_to_cluster_training_samples_soa.to_device(kd_tree_device.learning_to_cluster);
		kd_tree_device.learning_to_cluster.training_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_learning_to_cluster_training_sample_count);
		kd_tree_device.learning_to_cluster.training_sample_capacity = static_cast<unsigned int>(m_learning_to_cluster_training_samples.size());
	}

	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_light_clustering_count;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_normal_clustering_set_count;
	DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample> m_learning_to_cluster_training_samples;
	IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHost<DataContainer> m_learning_to_cluster_training_samples_soa;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_learning_to_cluster_training_sample_count;

	DataContainer<unsigned int> m_initial_light_cut_node_indices;
	DataContainer<IlluminationAwareKDTreeNormalClusteringSet> m_normal_clustering_sets;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_normal_face_observation_counts;
	DataContainer<unsigned int> m_light_cluster_node_indices;
	DataContainer<IlluminationAwareKDTreeLightClusterStatistics> m_light_cluster_statistics;
	DataContainer<unsigned short int> m_light_cluster_cdfs;
	DataContainer<IlluminationAwareKDTreeLightClusteringData> m_light_clustering_data;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_light_cluster_sample_counts;
	DataContainer<IlluminationAwareKDTreeSGShadingContext> m_representative_shading_contexts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_shading_context_states;

	unsigned int m_maximum_light_cut_size = LearningToClusterMaximumLightCutSize;
};

#endif
