/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using IlluminationAwareKDTreeDataHostInternal = GenericSoA<DataContainer, IlluminationAwareKDTreeNode, IlluminationAwareKDTreeNodeBounds>;

enum IlluminationAwareKDTreeDataHostBuffers
{
	ILLUMINATION_AWARE_KD_TREE_NODES,
	ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS
};

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	static constexpr unsigned int MAXIMUM_NUMBER_OF_NODES = 100000;

	void resize(unsigned int new_node_capacity, unsigned int new_training_sample_capacity, int new_tree_cut_size)
	{
		m_nodes_and_bounds.resize(new_node_capacity);

		GenericSoAHelpers::resize<DataContainer>(m_node_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_active_guiding_node_count, 1);
		GenericSoAHelpers::resize<DataContainer>(m_needs_split, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_guiding_distribution_count, 1);
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

		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_probabilities, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_cdfs, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_estimated_second_moment, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_effective_sample_count, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_batch_second_moment_sum, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_batch_sample_count, new_node_capacity * new_tree_cut_size);

		reset();
	}

	void reset()
	{
		// All reset is already done by the render pass
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_nodes_and_bounds.free();
		m_node_count					  = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_active_guiding_nodes			  = DataContainer<unsigned int>();
		m_active_guiding_node_count		  = DataContainer<unsigned int>();
		m_needs_split					  = DataContainer<uint8_t>();
		m_guiding_distribution_count	  = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_current_frontier				  = DataContainer<unsigned int>();
		m_current_frontier_count		  = DataContainer<unsigned int>();
		m_next_frontier					  = DataContainer<unsigned int>();
		m_next_frontier_count			  = DataContainer<unsigned int>();
		m_training_samples				  = DataContainer<IlluminationAwareKDTreeDirectIlluminationTrainingSample>();
		m_training_sample_count			  = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_batch_signatures				  = DataContainer<IlluminationAwareKDTreeIlluminationSignature>();
		m_history_signatures			  = DataContainer<IlluminationAwareKDTreeIlluminationSignature>();
		m_batch_spatial_moments			  = DataContainer<IlluminationAwareKDTreeSpatialSampleMoments>();
		m_history_spatial_moments		  = DataContainer<IlluminationAwareKDTreeSpatialSampleMoments>();
		m_tree_cut_sampling_probabilities = DataContainer<float>();
		m_tree_cut_sampling_cdfs		  = DataContainer<float>();
		m_estimated_second_moment		  = DataContainer<float>();
		m_effective_sample_count		  = DataContainer<float>();
		m_batch_second_moment_sum		  = DataContainer<float>();
		m_batch_sample_count			  = DataContainer<uint32_t>();

		return true;
	}

	std::size_t get_byte_size() const
	{
		return m_nodes_and_bounds.get_byte_size() + GenericSoAHelpers::get_byte_size(m_node_count) + GenericSoAHelpers::get_byte_size(m_active_guiding_nodes) +
			   GenericSoAHelpers::get_byte_size(m_active_guiding_node_count) + GenericSoAHelpers::get_byte_size(m_needs_split) +
			   GenericSoAHelpers::get_byte_size(m_guiding_distribution_count) + GenericSoAHelpers::get_byte_size(m_current_frontier) +
			   GenericSoAHelpers::get_byte_size(m_current_frontier_count) + GenericSoAHelpers::get_byte_size(m_next_frontier) +
			   GenericSoAHelpers::get_byte_size(m_next_frontier_count) + GenericSoAHelpers::get_byte_size(m_training_samples) +
			   GenericSoAHelpers::get_byte_size(m_training_sample_count) + GenericSoAHelpers::get_byte_size(m_batch_signatures) +
			   GenericSoAHelpers::get_byte_size(m_history_signatures) + GenericSoAHelpers::get_byte_size(m_batch_spatial_moments) +
			   GenericSoAHelpers::get_byte_size(m_history_spatial_moments) + GenericSoAHelpers::get_byte_size(m_tree_cut_sampling_probabilities) +
			   GenericSoAHelpers::get_byte_size(m_tree_cut_sampling_cdfs) + GenericSoAHelpers::get_byte_size(m_estimated_second_moment) +
			   GenericSoAHelpers::get_byte_size(m_effective_sample_count) + GenericSoAHelpers::get_byte_size(m_batch_second_moment_sum) +
			   GenericSoAHelpers::get_byte_size(m_batch_sample_count);
	}

	std::size_t maximum_size() const
	{
		return m_nodes_and_bounds.maximum_size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice device;

		device.user_settings = render_data.illumination_aware_kd_tree.user_settings;

		device.nodes				= m_nodes_and_bounds.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_NODES>();
		device.node_bounds			= m_nodes_and_bounds.template get_buffer_data_ptr<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>();
		device.node_capacity		= static_cast<unsigned int>(maximum_size());
		device.active_guiding_nodes = m_active_guiding_nodes.data();
		device.needs_split			= m_needs_split.data();
		device.current_frontier		= m_current_frontier.data();
		device.next_frontier		= m_next_frontier.data();

		if constexpr (std::is_same_v<DataContainer<GenericAtomicType<unsigned int, DataContainer>>, std::vector<std::atomic<unsigned int>>>)
		{
			device.active_guiding_node_count = m_active_guiding_node_count.data();

			device.node_count				  = m_node_count.data();
			device.current_frontier_count	  = m_current_frontier_count.data();
			device.next_frontier_count		  = m_next_frontier_count.data();
			device.guiding_distribution_count = m_guiding_distribution_count.data();
		}
		else
		{
			device.active_guiding_node_count = m_active_guiding_node_count.get_atomic_device_pointer();

			device.node_count				  = m_node_count.get_atomic_device_pointer();
			device.current_frontier_count	  = m_current_frontier_count.get_atomic_device_pointer();
			device.next_frontier_count		  = m_next_frontier_count.get_atomic_device_pointer();
			device.guiding_distribution_count = m_guiding_distribution_count.get_atomic_device_pointer();
		}
		device.training_samples			= m_training_samples.data();
		device.training_sample_capacity = static_cast<unsigned int>(m_training_samples.size());

		if constexpr (std::is_same_v<DataContainer<std::atomic<unsigned int>>, std::vector<std::atomic<unsigned int>>>)
			device.training_sample_count = m_training_sample_count.data();
		else
			device.training_sample_count = m_training_sample_count.get_atomic_device_pointer();

		device.batch_signatures		   = m_batch_signatures.data();
		device.history_signatures	   = m_history_signatures.data();
		device.batch_spatial_moments   = m_batch_spatial_moments.data();
		device.history_spatial_moments = m_history_spatial_moments.data();

		device.tree_cut_sampling_probabilities = m_tree_cut_sampling_probabilities.data();
		device.tree_cut_sampling_cdfs		   = m_tree_cut_sampling_cdfs.data();
		device.estimated_second_moment		   = m_estimated_second_moment.data();
		device.effective_sample_count		   = m_effective_sample_count.data();
		device.batch_second_moment_sum		   = m_batch_second_moment_sum.data();
		device.batch_sample_count			   = m_batch_sample_count.data();

		return device;
	}

	IlluminationAwareKDTreeDataHostInternal<DataContainer> m_nodes_and_bounds;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_node_count;

	DataContainer<unsigned int> m_active_guiding_nodes;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_active_guiding_node_count;
	DataContainer<uint8_t> m_needs_split;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_guiding_distribution_count;

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

	// Buffers below that point are for learning NEE distributions per each guiding cell
	DataContainer<float> m_tree_cut_sampling_probabilities;
	DataContainer<float> m_tree_cut_sampling_cdfs;
	DataContainer<float> m_estimated_second_moment;
	DataContainer<float> m_effective_sample_count;
	DataContainer<float> m_batch_second_moment_sum;
	DataContainer<uint32_t> m_batch_sample_count;
};

#endif
