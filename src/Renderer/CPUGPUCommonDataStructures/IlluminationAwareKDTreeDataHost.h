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
struct IlluminationAwareKDTreeDataHost
{
	static constexpr unsigned int MAXIMUM_NUMBER_OF_NODES = 100000;

	void resize(unsigned int new_node_capacity, unsigned int new_training_sample_capacity, int new_tree_cut_size)
	{
		GenericSoAHelpers::resize<DataContainer>(m_nodes, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_node_bounds, new_node_capacity);
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
		GenericSoAHelpers::resize<DataContainer>(m_nee_training_records, new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_nee_training_record_count, 1);

		GenericSoAHelpers::resize<DataContainer>(m_batch_signatures, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_signatures, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_batch_spatial_moments, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_spatial_moments, new_node_capacity);

		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_probabilities, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_cdfs, new_node_capacity * new_tree_cut_size);

		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_sample_count, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_sum_x, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_sum_y, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_sum_z, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_count, new_node_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cut_node_estimated_second_moment, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cut_node_sample_count, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_batch_per_cut_node_second_moment_sum, new_node_capacity * new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_batch_per_cut_node_sample_count, new_node_capacity * new_tree_cut_size);

		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_prior_pdfs, new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_prior_cdfs, new_tree_cut_size);
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

		m_needs_split				 = DataContainer<uint8_t>();
		m_guiding_distribution_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_current_frontier		 = DataContainer<unsigned int>();
		m_current_frontier_count = DataContainer<unsigned int>();
		m_next_frontier			 = DataContainer<unsigned int>();
		m_next_frontier_count	 = DataContainer<unsigned int>();

		m_training_samples			= DataContainer<IlluminationAwareKDTreeDirectIlluminationTrainingSample>();
		m_training_sample_count		= DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_nee_training_records		= DataContainer<IlluminationAwareKDTreeNEEDistributionTrainingRecord>();
		m_nee_training_record_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

		m_batch_signatures		  = DataContainer<IlluminationAwareKDTreeIlluminationSignature>();
		m_history_signatures	  = DataContainer<IlluminationAwareKDTreeIlluminationSignature>();
		m_batch_spatial_moments	  = DataContainer<IlluminationAwareKDTreeSpatialSampleMoments>();
		m_history_spatial_moments = DataContainer<IlluminationAwareKDTreeSpatialSampleMoments>();

		m_tree_cut_sampling_probabilities = DataContainer<unsigned short int>();
		m_tree_cut_sampling_cdfs		  = DataContainer<unsigned short int>();

		m_history_per_cell_sample_count				   = DataContainer<unsigned int>();
		m_history_per_cell_normal_sum_x				   = DataContainer<GenericAtomicType<float, DataContainer>>();
		m_history_per_cell_normal_sum_y				   = DataContainer<GenericAtomicType<float, DataContainer>>();
		m_history_per_cell_normal_sum_z				   = DataContainer<GenericAtomicType<float, DataContainer>>();
		m_history_per_cell_normal_count				   = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();
		m_history_per_cut_node_estimated_second_moment = DataContainer<float>();
		m_history_per_cut_node_sample_count			   = DataContainer<unsigned int>();
		m_batch_per_cut_node_second_moment_sum		   = DataContainer<float>();
		m_batch_per_cut_node_sample_count			   = DataContainer<unsigned int>();

		m_tree_cut_sampling_prior_pdfs = DataContainer<unsigned short int>();
		m_tree_cut_sampling_prior_cdfs = DataContainer<unsigned short int>();

		return true;
	}

	std::size_t maximum_size() const
	{
		return m_nodes.size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice device;

		device.user_settings								  = render_data.illumination_aware_kd_tree.user_settings;
		device.nee_learnt_distributions.learning_nee_settings = render_data.illumination_aware_kd_tree.nee_learnt_distributions.learning_nee_settings;

		device.nodes		 = GenericSoAHelpers::get_buffer_data_ptr(m_nodes);
		device.node_bounds	 = GenericSoAHelpers::get_buffer_data_ptr(m_node_bounds);
		device.node_capacity = static_cast<unsigned int>(maximum_size());

		device.active_guiding_nodes = GenericSoAHelpers::get_buffer_data_ptr(m_active_guiding_nodes);
		device.needs_split			= GenericSoAHelpers::get_buffer_data_ptr(m_needs_split);
		device.current_frontier		= GenericSoAHelpers::get_buffer_data_ptr(m_current_frontier);
		device.next_frontier		= GenericSoAHelpers::get_buffer_data_ptr(m_next_frontier);

		device.active_guiding_node_count  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_active_guiding_node_count);
		device.node_count				  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_node_count);
		device.current_frontier_count	  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_current_frontier_count);
		device.next_frontier_count		  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_next_frontier_count);
		device.guiding_distribution_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_guiding_distribution_count);

		device.training_samples										 = GenericSoAHelpers::get_buffer_data_ptr(m_training_samples);
		device.training_sample_capacity								 = static_cast<unsigned int>(m_training_samples.size());
		device.nee_learnt_distributions.nee_training_records		 = GenericSoAHelpers::get_buffer_data_ptr(m_nee_training_records);
		device.nee_learnt_distributions.nee_training_record_capacity = static_cast<unsigned int>(m_nee_training_records.size());

		device.training_sample_count							  = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_training_sample_count);
		device.nee_learnt_distributions.nee_training_record_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_nee_training_record_count);

		device.batch_signatures		   = GenericSoAHelpers::get_buffer_data_ptr(m_batch_signatures);
		device.history_signatures	   = GenericSoAHelpers::get_buffer_data_ptr(m_history_signatures);
		device.batch_spatial_moments   = GenericSoAHelpers::get_buffer_data_ptr(m_batch_spatial_moments);
		device.history_spatial_moments = GenericSoAHelpers::get_buffer_data_ptr(m_history_spatial_moments);

		device.nee_learnt_distributions.tree_cut_sampling_probabilities = GenericSoAHelpers::get_buffer_data_ptr(m_tree_cut_sampling_probabilities);
		device.nee_learnt_distributions.tree_cut_sampling_cdfs			= GenericSoAHelpers::get_buffer_data_ptr(m_tree_cut_sampling_cdfs);

		device.nee_learnt_distributions.history_per_cell_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cell_sample_count);
		device.nee_learnt_distributions.history_per_cell_normal_sum_x = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cell_normal_sum_x);
		device.nee_learnt_distributions.history_per_cell_normal_sum_y = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cell_normal_sum_y);
		device.nee_learnt_distributions.history_per_cell_normal_sum_z = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cell_normal_sum_z);
		device.nee_learnt_distributions.history_per_cell_normal_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cell_normal_count);
		device.nee_learnt_distributions.history_per_cut_node_estimated_second_moment =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cut_node_estimated_second_moment);
		device.nee_learnt_distributions.history_per_cut_node_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_history_per_cut_node_sample_count);
		device.nee_learnt_distributions.batch_per_cut_node_second_moment_sum =
			GenericSoAHelpers::get_buffer_data_atomic_ptr(m_batch_per_cut_node_second_moment_sum);
		device.nee_learnt_distributions.batch_per_cut_node_sample_count = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_batch_per_cut_node_sample_count);

		device.nee_learnt_distributions.tree_cut_sampling_prior_pdfs = GenericSoAHelpers::get_buffer_data_ptr(m_tree_cut_sampling_prior_pdfs);
		device.nee_learnt_distributions.tree_cut_sampling_prior_cdfs = GenericSoAHelpers::get_buffer_data_ptr(m_tree_cut_sampling_prior_cdfs);

		return device;
	}

	DataContainer<IlluminationAwareKDTreeNode> m_nodes;
	DataContainer<IlluminationAwareKDTreeNodeBounds> m_node_bounds;
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
	DataContainer<IlluminationAwareKDTreeNEEDistributionTrainingRecord> m_nee_training_records;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_nee_training_record_count;

	DataContainer<IlluminationAwareKDTreeIlluminationSignature> m_batch_signatures;
	DataContainer<IlluminationAwareKDTreeIlluminationSignature> m_history_signatures;
	DataContainer<IlluminationAwareKDTreeSpatialSampleMoments> m_batch_spatial_moments;
	DataContainer<IlluminationAwareKDTreeSpatialSampleMoments> m_history_spatial_moments;

	// Buffers below that point are for learning NEE distributions per each guiding cell
	DataContainer<unsigned short int> m_tree_cut_sampling_probabilities;
	DataContainer<unsigned short int> m_tree_cut_sampling_cdfs;

	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_history_per_cell_sample_count;
	DataContainer<GenericAtomicType<float, DataContainer>> m_history_per_cell_normal_sum_x;
	DataContainer<GenericAtomicType<float, DataContainer>> m_history_per_cell_normal_sum_y;
	DataContainer<GenericAtomicType<float, DataContainer>> m_history_per_cell_normal_sum_z;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_history_per_cell_normal_count;
	DataContainer<GenericAtomicType<float, DataContainer>> m_history_per_cut_node_estimated_second_moment;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_history_per_cut_node_sample_count;
	DataContainer<GenericAtomicType<float, DataContainer>> m_batch_per_cut_node_second_moment_sum;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_batch_per_cut_node_sample_count;

	DataContainer<unsigned short int> m_tree_cut_sampling_prior_pdfs;
	DataContainer<unsigned short int> m_tree_cut_sampling_prior_cdfs;
};

#endif
