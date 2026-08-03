/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeIlluminationSignatureSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeSpatialSampleMomentsSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	static constexpr unsigned int MAXIMUM_NUMBER_OF_NODES = 100000;

	void resize(unsigned int new_node_capacity, unsigned int new_training_sample_capacity)
	{
		GenericSoAHelpers::resize<DataContainer>(m_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_node_bounds, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_node_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_node_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_needs_split, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_current_frontier, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_current_frontier_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_next_frontier, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_next_frontier_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_training_samples, new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_training_sample_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_samples, new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_sample_count, 1);

		m_batch_signatures.resize(new_node_capacity);
		m_history_signatures.resize(new_node_capacity);
		m_batch_spatial_moments.resize(new_node_capacity);
		m_history_spatial_moments.resize(new_node_capacity);

		size_t cluster_slot_capacity = static_cast<size_t>(new_node_capacity) * IlluminationAwareKDTreeMaximumLightCutSize;
		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_node_indices, cluster_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_statistics, cluster_slot_capacity);
		m_light_cluster_batch_statistics.resize(static_cast<unsigned int>(cluster_slot_capacity));
		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_data, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_batch_sample_counts, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_shading_contexts, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_shading_context_states, new_node_capacity);

		GenericSoAHelpers::resize<DataContainer>(m_initial_light_cut_node_indices, IlluminationAwareKDTreeMaximumLightCutSize);
	}

	void reset()
	{
		// All reset is already done by the render pass
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_nodes		  = DataContainer<IlluminationAwareKDTreeNode>();
		m_node_bounds = DataContainer<IlluminationAwareKDTreeNodeBounds>();
		m_node_count  = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_active_guiding_nodes		= DataContainer<unsigned int>();
		m_active_guiding_node_count = DataContainer<unsigned int>();

		m_needs_split			 = DataContainer<unsigned char>();
		m_light_clustering_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_current_frontier		 = DataContainer<unsigned int>();
		m_current_frontier_count = DataContainer<unsigned int>();
		m_next_frontier			 = DataContainer<unsigned int>();
		m_next_frontier_count	 = DataContainer<unsigned int>();

		m_training_samples							= DataContainer<IlluminationAwareKDTreeDirectIlluminationTrainingSample>();
		m_training_sample_count						= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_learning_to_cluster_training_samples		= DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample>();
		m_learning_to_cluster_training_sample_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_batch_signatures.free();
		m_history_signatures.free();
		m_batch_spatial_moments.free();
		m_history_spatial_moments.free();

		m_light_cluster_node_indices = DataContainer<unsigned int>();
		m_light_cluster_statistics	 = DataContainer<IlluminationAwareKDTreeLightClusterStatistics>();
		m_light_cluster_batch_statistics.free();
		m_light_clustering_data					= DataContainer<IlluminationAwareKDTreeLightClusteringData>();
		m_light_clustering_batch_sample_counts	= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_representative_shading_contexts		= DataContainer<IlluminationAwareKDTreeSGShadingContext>();
		m_representative_shading_context_states = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_initial_light_cut_node_indices		= DataContainer<unsigned int>();

		return true;
	}

	std::size_t maximum_size() const
	{
		return m_nodes.size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice device;

		device.user_settings					 = render_data.illumination_aware_kd_tree.user_settings;
		device.learning_to_cluster.user_settings = render_data.illumination_aware_kd_tree.learning_to_cluster.user_settings;

		device.nodes		 = GenericSoAHelpers::get_buffer_data_ptr(m_nodes);
		device.node_bounds	 = GenericSoAHelpers::get_buffer_data_ptr(m_node_bounds);
		device.node_capacity = static_cast<unsigned int>(maximum_size());

		device.active_guiding_nodes = GenericSoAHelpers::get_buffer_data_ptr(m_active_guiding_nodes);
		device.needs_split			= GenericSoAHelpers::get_buffer_data_ptr(m_needs_split);
		device.current_frontier		= GenericSoAHelpers::get_buffer_data_ptr(m_current_frontier);
		device.next_frontier		= GenericSoAHelpers::get_buffer_data_ptr(m_next_frontier);

		device.active_guiding_node_count				  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_active_guiding_node_count);
		device.node_count								  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_node_count);
		device.current_frontier_count					  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_current_frontier_count);
		device.next_frontier_count						  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_next_frontier_count);
		device.learning_to_cluster.light_clustering_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_light_clustering_count);

		device.learning_to_cluster.initial_light_cut_node_indices = GenericSoAHelpers::get_buffer_data_ptr(m_initial_light_cut_node_indices);

		device.learning_to_cluster.light_cluster_node_indices			= GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_node_indices);
		device.learning_to_cluster.light_cluster_statistics				= GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_statistics);
		device.learning_to_cluster.light_cluster_batch_statistics		= m_light_cluster_batch_statistics.to_device();
		device.learning_to_cluster.light_clustering_data				= GenericSoAHelpers::get_buffer_data_ptr(m_light_clustering_data);
		device.learning_to_cluster.light_clustering_batch_sample_counts = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_light_clustering_batch_sample_counts);
		device.learning_to_cluster.representative_shading_contexts		= GenericSoAHelpers::get_buffer_data_ptr(m_representative_shading_contexts);
		device.learning_to_cluster.representative_shading_context_states =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_shading_context_states);

		device.training_samples			= GenericSoAHelpers::get_buffer_data_ptr(m_training_samples);
		device.training_sample_capacity = static_cast<unsigned int>(m_training_samples.size());

		device.training_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_training_sample_count);

		device.learning_to_cluster_training_samples			= GenericSoAHelpers::get_buffer_data_ptr(m_learning_to_cluster_training_samples);
		device.learning_to_cluster_training_sample_capacity = static_cast<unsigned int>(m_learning_to_cluster_training_samples.size());
		device.learning_to_cluster_training_sample_count	= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_learning_to_cluster_training_sample_count);

		device.batch_signatures		   = m_batch_signatures.to_device();
		device.history_signatures	   = m_history_signatures.to_device();
		device.batch_spatial_moments   = m_batch_spatial_moments.to_device();
		device.history_spatial_moments = m_history_spatial_moments.to_device();

		return device;
	}

	DataContainer<IlluminationAwareKDTreeNode> m_nodes;
	DataContainer<IlluminationAwareKDTreeNodeBounds> m_node_bounds;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_node_count;

	DataContainer<unsigned int> m_active_guiding_nodes;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_active_guiding_node_count;
	DataContainer<unsigned char> m_needs_split;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_light_clustering_count;

	DataContainer<unsigned int> m_current_frontier;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_current_frontier_count;
	DataContainer<unsigned int> m_next_frontier;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_next_frontier_count;

	DataContainer<IlluminationAwareKDTreeDirectIlluminationTrainingSample> m_training_samples;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_training_sample_count;
	DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample> m_learning_to_cluster_training_samples;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_learning_to_cluster_training_sample_count;

	IlluminationAwareKDTreeIlluminationSignatureSoAHost<DataContainer> m_batch_signatures;
	IlluminationAwareKDTreeIlluminationSignatureSoAHost<DataContainer> m_history_signatures;
	IlluminationAwareKDTreeSpatialSampleMomentsSoAHost<DataContainer> m_batch_spatial_moments;
	IlluminationAwareKDTreeSpatialSampleMomentsSoAHost<DataContainer> m_history_spatial_moments;

	DataContainer<unsigned int> m_initial_light_cut_node_indices;

	DataContainer<unsigned int> m_light_cluster_node_indices;
	DataContainer<IlluminationAwareKDTreeLightClusterStatistics> m_light_cluster_statistics;
	IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost<DataContainer> m_light_cluster_batch_statistics;
	DataContainer<IlluminationAwareKDTreeLightClusteringData> m_light_clustering_data;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_light_clustering_batch_sample_counts;
	DataContainer<IlluminationAwareKDTreeSGShadingContext> m_representative_shading_contexts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_shading_context_states;
};

#endif
