/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeCoreDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeIlluminationSignatureSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeNISMLDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeSpatialSampleMomentsSoAHost.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	void resize(unsigned int new_node_capacity,
				unsigned int new_training_sample_capacity,
				unsigned int new_nisml_representative_capacity	 = 1,
				unsigned int new_nisml_hash_table_reserved_bytes = 100000000u,
				unsigned int new_nisml_hash_normal_precision	 = 2u)
	{
		m_kd_tree_data.resize(new_node_capacity, new_training_sample_capacity);
		m_nisml_data.resize(new_node_capacity, new_nisml_representative_capacity, new_nisml_hash_table_reserved_bytes, new_nisml_hash_normal_precision);

		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_normal_clustering_set_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_samples, new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_learning_to_cluster_training_sample_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_initial_light_cut_node_indices, LearningToClusterMaximumLightCutSize);
		GenericSoAHelpers::resize<DataContainer>(m_normal_clustering_sets, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_normal_face_observation_counts, static_cast<size_t>(new_node_capacity) * SurfaceNormalFace_Count);

		size_t light_clustering_capacity   = static_cast<size_t>(new_node_capacity) * 2;
		size_t light_cluster_slot_capacity = light_clustering_capacity * LearningToClusterMaximumLightCutSize;
		size_t pending_record_capacity	   = light_clustering_capacity * IlluminationAwareKDTreePendingLightClusterRecordStride;

		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_node_indices, light_cluster_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_cluster_statistics, light_cluster_slot_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_light_clustering_data, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_pending_light_cluster_records, pending_record_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_pending_light_cluster_record_counts, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_reservoir_seen_counts, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_reservoir_proposals, pending_record_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_shading_contexts, light_clustering_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_representative_shading_context_states, light_clustering_capacity);

		GenericSoAHelpers::resize<DataContainer>(m_any_cell_needs_split, 1);
		GenericSoAHelpers::resize_host_pinned_mem(m_any_cell_needs_split_host_pinned, 1);
	}

	void reset()
	{
		// All reset is already done by the render pass
	}

	bool free()
	{
		bool core_data_freed			 = m_kd_tree_data.free();
		bool nisml_data_freed			 = m_nisml_data.free();
		bool light_clustering_data_freed = m_light_clustering_count.size() > 0;

		m_light_clustering_count					= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_normal_clustering_set_count				= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_learning_to_cluster_training_samples		= DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample>();
		m_learning_to_cluster_training_sample_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_initial_light_cut_node_indices			= DataContainer<unsigned int>();
		m_normal_clustering_sets					= DataContainer<IlluminationAwareKDTreeNormalClusteringSet>();
		m_normal_face_observation_counts			= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_light_cluster_node_indices				= DataContainer<unsigned int>();
		m_light_cluster_statistics					= DataContainer<IlluminationAwareKDTreeLightClusterStatistics>();
		m_light_clustering_data						= DataContainer<IlluminationAwareKDTreeLightClusteringData>();
		m_pending_light_cluster_records				= DataContainer<IlluminationAwareKDTreePendingLightClusterRecord>();
		m_pending_light_cluster_record_counts		= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_reservoir_seen_counts						= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_reservoir_proposals						= DataContainer<GenericAtomicType<unsigned long long int, DataContainer>>();
		m_representative_shading_contexts			= DataContainer<IlluminationAwareKDTreeSGShadingContext>();
		m_representative_shading_context_states		= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		bool any_cell_needs_split_freed = m_any_cell_needs_split.size() > 0;
		m_any_cell_needs_split			= DataContainer<unsigned char>();

		bool any_cell_needs_split_host_pinned_freed = m_any_cell_needs_split_host_pinned.size() > 0;
		m_any_cell_needs_split_host_pinned			= DataContainer<unsigned char>();

		return core_data_freed || nisml_data_freed || light_clustering_data_freed || any_cell_needs_split_freed || any_cell_needs_split_host_pinned_freed;
	}

	std::size_t maximum_size() const
	{
		return m_kd_tree_data.maximum_size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice kd_tree_device = m_kd_tree_data.to_device();

		m_nisml_data.to_device(kd_tree_device);
		kd_tree_device.any_cell_needs_split = GenericSoAHelpers::get_buffer_data_ptr(m_any_cell_needs_split);

		kd_tree_device.core.user_settings									= render_data.kd_tree_device.core.user_settings;
		kd_tree_device.learning_to_cluster.user_settings					= render_data.kd_tree_device.learning_to_cluster.user_settings;
		kd_tree_device.learning_to_cluster.effective_initial_light_cut_size = render_data.kd_tree_device.learning_to_cluster.effective_initial_light_cut_size;

		kd_tree_device.learning_to_cluster.light_clustering_count			= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_light_clustering_count);
		kd_tree_device.learning_to_cluster.light_clustering_capacity		= static_cast<unsigned int>(m_light_clustering_data.size());
		kd_tree_device.learning_to_cluster.normal_clustering_sets			= GenericSoAHelpers::get_buffer_data_ptr(m_normal_clustering_sets);
		kd_tree_device.learning_to_cluster.normal_clustering_set_count		= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_normal_clustering_set_count);
		kd_tree_device.learning_to_cluster.normal_clustering_set_capacity	= static_cast<unsigned int>(m_normal_clustering_sets.size());
		kd_tree_device.learning_to_cluster.normal_face_observation_counts	= GenericSoAHelpers::get_buffer_data_atomic_ptr(m_normal_face_observation_counts);
		kd_tree_device.learning_to_cluster.initial_light_cut_node_indices	= GenericSoAHelpers::get_buffer_data_ptr(m_initial_light_cut_node_indices);
		kd_tree_device.learning_to_cluster.effective_initial_light_cut_size = render_data.kd_tree_device.learning_to_cluster.effective_initial_light_cut_size;
		kd_tree_device.learning_to_cluster.light_cluster_node_indices		= GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_node_indices);
		kd_tree_device.learning_to_cluster.light_cluster_statistics			= GenericSoAHelpers::get_buffer_data_ptr(m_light_cluster_statistics);
		kd_tree_device.learning_to_cluster.light_clustering_data			= GenericSoAHelpers::get_buffer_data_ptr(m_light_clustering_data);
		kd_tree_device.learning_to_cluster.pending_light_cluster_records	= GenericSoAHelpers::get_buffer_data_ptr(m_pending_light_cluster_records);
		kd_tree_device.learning_to_cluster.pending_light_cluster_record_counts =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_pending_light_cluster_record_counts);
		kd_tree_device.learning_to_cluster.reservoir_seen_counts		   = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_reservoir_seen_counts);
		kd_tree_device.learning_to_cluster.reservoir_proposals			   = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_reservoir_proposals);
		kd_tree_device.learning_to_cluster.pending_record_stride		   = IlluminationAwareKDTreePendingLightClusterRecordStride;
		kd_tree_device.learning_to_cluster.representative_shading_contexts = GenericSoAHelpers::get_buffer_data_ptr(m_representative_shading_contexts);
		kd_tree_device.learning_to_cluster.representative_shading_context_states =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_representative_shading_context_states);

		kd_tree_device.learning_to_cluster_training_samples		 = GenericSoAHelpers::get_buffer_data_ptr(m_learning_to_cluster_training_samples);
		kd_tree_device.learning_to_cluster_training_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_learning_to_cluster_training_sample_count);
		kd_tree_device.learning_to_cluster_training_sample_capacity = static_cast<unsigned int>(m_learning_to_cluster_training_samples.size());

		return kd_tree_device;
	}

	IlluminationAwareKDTreeCoreDataHost<DataContainer> m_kd_tree_data;
	IlluminationAwareKDTreeNISMLDataHost<DataContainer> m_nisml_data;

	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_light_clustering_count;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_normal_clustering_set_count;
	DataContainer<IlluminationAwareKDTreeLearningToClusterTrainingSample> m_learning_to_cluster_training_samples;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_learning_to_cluster_training_sample_count;

	DataContainer<unsigned int> m_initial_light_cut_node_indices;
	DataContainer<IlluminationAwareKDTreeNormalClusteringSet> m_normal_clustering_sets;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_normal_face_observation_counts;
	DataContainer<unsigned int> m_light_cluster_node_indices;
	DataContainer<IlluminationAwareKDTreeLightClusterStatistics> m_light_cluster_statistics;
	DataContainer<IlluminationAwareKDTreeLightClusteringData> m_light_clustering_data;
	DataContainer<IlluminationAwareKDTreePendingLightClusterRecord> m_pending_light_cluster_records;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_pending_light_cluster_record_counts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_reservoir_seen_counts;
	DataContainer<GenericAtomicType<unsigned long long int, DataContainer>> m_reservoir_proposals;
	DataContainer<IlluminationAwareKDTreeSGShadingContext> m_representative_shading_contexts;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_representative_shading_context_states;

	DataContainer<unsigned char> m_any_cell_needs_split;
	DataContainer<unsigned char> m_any_cell_needs_split_host_pinned;
};

#endif
