/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_CORE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_CORE_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeCoreDataHost
{
	static constexpr unsigned int MAXIMUM_NUMBER_OF_NODES				  = 200000;
	static constexpr unsigned int INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY = 2000000;

	void resize(unsigned int new_node_capacity, unsigned int new_training_sample_capacity)
	{
		GenericSoAHelpers::resize<DataContainer>(m_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_node_bounds, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_parent_indices, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_node_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_node_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_needs_split, new_node_capacity);

		GenericSoAHelpers::resize<DataContainer>(m_current_frontier, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_current_frontier_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_next_frontier, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_next_frontier_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_training_samples, new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_training_sample_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_batch_signatures, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_signatures, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_batch_spatial_moments, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_spatial_moments, new_node_capacity);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_nodes			 = DataContainer<IlluminationAwareKDTreeNode>();
		m_node_bounds	 = DataContainer<IlluminationAwareKDTreeNodeBounds>();
		m_parent_indices = DataContainer<unsigned int>();
		m_node_count	 = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_active_guiding_nodes		= DataContainer<unsigned int>();
		m_active_guiding_node_count = DataContainer<unsigned int>();

		m_needs_split = DataContainer<uint8_t>();

		m_current_frontier		 = DataContainer<unsigned int>();
		m_current_frontier_count = DataContainer<unsigned int>();
		m_next_frontier			 = DataContainer<unsigned int>();
		m_next_frontier_count	 = DataContainer<unsigned int>();

		m_training_samples		= DataContainer<IlluminationAwareKDTreeDirectIlluminationTrainingSample>();
		m_training_sample_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_batch_signatures		  = DataContainer<IlluminationAwareKDTreeIlluminationSignature>();
		m_history_signatures	  = DataContainer<IlluminationAwareKDTreeIlluminationSignature>();
		m_batch_spatial_moments	  = DataContainer<IlluminationAwareKDTreeSpatialSampleMoments>();
		m_history_spatial_moments = DataContainer<IlluminationAwareKDTreeSpatialSampleMoments>();

		return true;
	}

	std::size_t maximum_size() const
	{
		return m_nodes.size();
	}

	IlluminationAwareKDTreeDevice to_device()
	{
		IlluminationAwareKDTreeDevice kd_tree_device;

		kd_tree_device.core.nodes		   = GenericSoAHelpers::get_buffer_data_ptr(m_nodes);
		kd_tree_device.core.node_bounds	   = GenericSoAHelpers::get_buffer_data_ptr(m_node_bounds);
		kd_tree_device.core.parent_indices = GenericSoAHelpers::get_buffer_data_ptr(m_parent_indices);
		kd_tree_device.core.node_capacity  = static_cast<unsigned int>(maximum_size());

		kd_tree_device.core.active_guiding_nodes = GenericSoAHelpers::get_buffer_data_ptr(m_active_guiding_nodes);
		kd_tree_device.core.needs_split			 = GenericSoAHelpers::get_buffer_data_ptr(m_needs_split);
		kd_tree_device.core.current_frontier	 = GenericSoAHelpers::get_buffer_data_ptr(m_current_frontier);
		kd_tree_device.core.next_frontier		 = GenericSoAHelpers::get_buffer_data_ptr(m_next_frontier);

		kd_tree_device.core.active_guiding_node_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_active_guiding_node_count);
		kd_tree_device.core.node_count				  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_node_count);
		kd_tree_device.core.current_frontier_count	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_current_frontier_count);
		kd_tree_device.core.next_frontier_count		  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_next_frontier_count);

		kd_tree_device.core.training_samples		 = GenericSoAHelpers::get_buffer_data_ptr(m_training_samples);
		kd_tree_device.core.training_sample_capacity = static_cast<unsigned int>(m_training_samples.size());
		kd_tree_device.core.training_sample_count	 = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_training_sample_count);

		kd_tree_device.core.batch_signatures		= GenericSoAHelpers::get_buffer_data_ptr(m_batch_signatures);
		kd_tree_device.core.history_signatures		= GenericSoAHelpers::get_buffer_data_ptr(m_history_signatures);
		kd_tree_device.core.batch_spatial_moments	= GenericSoAHelpers::get_buffer_data_ptr(m_batch_spatial_moments);
		kd_tree_device.core.history_spatial_moments = GenericSoAHelpers::get_buffer_data_ptr(m_history_spatial_moments);

		return kd_tree_device;
	}

	DataContainer<IlluminationAwareKDTreeNode> m_nodes;
	DataContainer<IlluminationAwareKDTreeNodeBounds> m_node_bounds;
	DataContainer<unsigned int> m_parent_indices;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_node_count;

	DataContainer<unsigned int> m_active_guiding_nodes;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_active_guiding_node_count;
	DataContainer<uint8_t> m_needs_split;

	DataContainer<unsigned int> m_current_frontier;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_current_frontier_count;
	DataContainer<unsigned int> m_next_frontier;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_next_frontier_count;

	DataContainer<IlluminationAwareKDTreeDirectIlluminationTrainingSample> m_training_samples;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_training_sample_count;

	DataContainer<IlluminationAwareKDTreeIlluminationSignature> m_batch_signatures;
	DataContainer<IlluminationAwareKDTreeIlluminationSignature> m_history_signatures;
	DataContainer<IlluminationAwareKDTreeSpatialSampleMoments> m_batch_spatial_moments;
	DataContainer<IlluminationAwareKDTreeSpatialSampleMoments> m_history_spatial_moments;
};

#endif // #ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_CORE_DATA_HOST_H
