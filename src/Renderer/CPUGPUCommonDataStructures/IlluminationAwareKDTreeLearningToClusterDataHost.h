/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeLearningToClusterDataHost
{
	void resize(unsigned int new_node_capacity,
				unsigned int new_training_sample_capacity,
				unsigned int maximum_lightcut_size = LearningToClusterMaximumLightCutSize)
	{
		m_maximum_lightcut_size = maximum_lightcut_size;

		GenericSoAHelpers::resize<DataContainer>(m_lightcut_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_normal_lightcut_set_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_samples, new_training_sample_capacity);
		m_learning_to_cluster_training_samples_soa.resize(new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_sample_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_initial_lightcut_node_indices, maximum_lightcut_size);
		GenericSoAHelpers::resize<DataContainer>(m_normal_lightcut_sets, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_normal_face_observation_counts, static_cast<std::size_t>(new_node_capacity) * SurfaceNormalFace_Count);

		// New node capacity is essentially the node capacity of the illumination aware KD-tree. Clearly not all nodes of the KD-tree become guiding nodes
		// containing learning to cluster distributions. So we would think that this could be resized to much lower new_node_capacity. But because learning to
		// cluster allocates distributions per face-normal * mesh_id, we need more allocation capacity than just the number of guiding nodes of the KD-tree so
		// that's why we still allocate new node capacity here
		std::size_t lightcut_capacity	   = static_cast<std::size_t>(new_node_capacity);
		std::size_t lightcut_slot_capacity = lightcut_capacity * maximum_lightcut_size;

		GenericSoAHelpers::resize<DataContainer>(m_lightcut_node_indices, lightcut_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_lightcut_statistics, lightcut_slot_capacity);
		m_lightcut_batch_statistics.resize(static_cast<unsigned int>(lightcut_slot_capacity));
		GenericSoAHelpers::resize<DataContainer>(m_lightcut_cdfs, lightcut_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_lightcut_data, lightcut_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_lightcut_sample_counts, lightcut_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_lightcut_representative_shading_contexts, lightcut_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_lightcut_representative_shading_context_states, lightcut_capacity);
	}

	bool free()
	{
		bool lightcut_data_freed = m_lightcut_count.size() > 0;

		m_lightcut_count									= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_normal_lightcut_set_count							= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_learning_to_cluster_training_samples				= DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample>();
		m_learning_to_cluster_training_sample_count			= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		bool learning_to_cluster_training_samples_soa_freed = m_learning_to_cluster_training_samples_soa.maximum_size() > 0;
		m_learning_to_cluster_training_samples_soa.free();
		m_initial_lightcut_node_indices		 = DataContainer<unsigned int>();
		m_normal_lightcut_sets				 = DataContainer<IlluminationAwareKDTreeLearningToClusterLightcutSet>();
		m_normal_face_observation_counts	 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_lightcut_node_indices				 = DataContainer<unsigned int>();
		m_lightcut_statistics				 = DataContainer<IlluminationAwareKDTreeLightClusterStatistics>();
		bool lightcut_batch_statistics_freed = m_lightcut_batch_statistics.maximum_size() > 0;
		m_lightcut_batch_statistics.free();
		m_lightcut_cdfs									 = DataContainer<unsigned short int>();
		m_lightcut_data									 = DataContainer<IlluminationAwareKDTreeLightClusteringData>();
		m_lightcut_sample_counts						 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_lightcut_representative_shading_contexts		 = DataContainer<IlluminationAwareKDTreeSGShadingContext>();
		m_lightcut_representative_shading_context_states = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		return lightcut_data_freed || learning_to_cluster_training_samples_soa_freed || lightcut_batch_statistics_freed;
	}

	std::size_t maximum_size() const
	{
		return m_lightcut_data.size();
	}

	void to_device(IlluminationAwareKDTreeDevice& kd_tree_device)
	{
		kd_tree_device.learning_to_cluster.lightcut_count				  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_lightcut_count);
		kd_tree_device.learning_to_cluster.lightcut_capacity			  = static_cast<unsigned int>(m_lightcut_data.size());
		kd_tree_device.learning_to_cluster.normal_lightcut_sets			  = GenericSoAHelpers::get_buffer_data_ptr(m_normal_lightcut_sets);
		kd_tree_device.learning_to_cluster.normal_lightcut_set_count	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_normal_lightcut_set_count);
		kd_tree_device.learning_to_cluster.normal_lightcut_set_capacity	  = static_cast<unsigned int>(m_normal_lightcut_sets.size());
		kd_tree_device.learning_to_cluster.normal_face_observation_counts = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_normal_face_observation_counts);
		kd_tree_device.learning_to_cluster.initial_lightcut_node_indices  = GenericSoAHelpers::get_buffer_data_ptr(m_initial_lightcut_node_indices);
		kd_tree_device.learning_to_cluster.lightcut_node_indices		  = GenericSoAHelpers::get_buffer_data_ptr(m_lightcut_node_indices);
		kd_tree_device.learning_to_cluster.lightcut_statistics			  = GenericSoAHelpers::get_buffer_data_ptr(m_lightcut_statistics);
		kd_tree_device.learning_to_cluster.lightcut_batch_statistics	  = m_lightcut_batch_statistics.to_device();
		kd_tree_device.learning_to_cluster.lightcut_cdfs				  = GenericSoAHelpers::get_buffer_data_ptr(m_lightcut_cdfs);
		kd_tree_device.learning_to_cluster.lightcut_data				  = GenericSoAHelpers::get_buffer_data_ptr(m_lightcut_data);
		kd_tree_device.learning_to_cluster.lightcut_sample_counts		  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_lightcut_sample_counts);
		kd_tree_device.learning_to_cluster.lightcut_representative_shading_contexts =
			GenericSoAHelpers::get_buffer_data_ptr(m_lightcut_representative_shading_contexts);
		kd_tree_device.learning_to_cluster.lightcut_representative_shading_context_states =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_lightcut_representative_shading_context_states);

		kd_tree_device.learning_to_cluster.training_samples = GenericSoAHelpers::get_buffer_data_ptr(m_learning_to_cluster_training_samples);
		m_learning_to_cluster_training_samples_soa.to_device(kd_tree_device.learning_to_cluster);
		kd_tree_device.learning_to_cluster.training_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_learning_to_cluster_training_sample_count);
		kd_tree_device.learning_to_cluster.training_sample_capacity = static_cast<unsigned int>(m_learning_to_cluster_training_samples.size());
	}

	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_lightcut_count;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_normal_lightcut_set_count;
	DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample> m_learning_to_cluster_training_samples;
	IlluminationAwareKDTreeLearningToClusterTrainingSampleSoAHost<DataContainer> m_learning_to_cluster_training_samples_soa;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_learning_to_cluster_training_sample_count;

	DataContainer<unsigned int> m_initial_lightcut_node_indices;
	DataContainer<IlluminationAwareKDTreeLearningToClusterLightcutSet> m_normal_lightcut_sets;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_normal_face_observation_counts;
	DataContainer<unsigned int> m_lightcut_node_indices;
	DataContainer<IlluminationAwareKDTreeLightClusterStatistics> m_lightcut_statistics;
	IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost<DataContainer> m_lightcut_batch_statistics;
	DataContainer<unsigned short int> m_lightcut_cdfs;
	DataContainer<IlluminationAwareKDTreeLightClusteringData> m_lightcut_data;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_lightcut_sample_counts;
	DataContainer<IlluminationAwareKDTreeSGShadingContext> m_lightcut_representative_shading_contexts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_lightcut_representative_shading_context_states;

	unsigned int m_maximum_lightcut_size = LearningToClusterMaximumLightCutSize;
};

#endif // #ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LEARNING_TO_CLUSTER_DATA_HOST_H
