/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_NEE_LEARNT_DISTRIBUTIONS_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_NEE_LEARNT_DISTRIBUTIONS_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeNEELearntDistributionsDataHost
{
	void resize(unsigned int new_node_capacity, unsigned int new_training_sample_capacity, int new_tree_cut_size)
	{
		GenericSoAHelpers::resize<DataContainer>(m_training_records, new_training_sample_capacity);
		GenericSoAHelpers::resize<DataContainer>(m_training_record_count, 1);

		unsigned int normal_face_distribution_count = new_node_capacity * static_cast<unsigned int>(SurfaceNormalFace_Count);
		unsigned int distribution_slot_count		= normal_face_distribution_count * new_tree_cut_size;
		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_probabilities, distribution_slot_count);
		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_cdfs, distribution_slot_count);

		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_sample_count, normal_face_distribution_count);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_sum_x, normal_face_distribution_count);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_sum_y, normal_face_distribution_count);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_sum_z, normal_face_distribution_count);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cell_normal_count, normal_face_distribution_count);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cut_node_estimated_second_moment, distribution_slot_count);
		GenericSoAHelpers::resize<DataContainer>(m_history_per_cut_node_sample_count, distribution_slot_count);
		GenericSoAHelpers::resize<DataContainer>(m_batch_per_cut_node_second_moment_sum, distribution_slot_count);
		GenericSoAHelpers::resize<DataContainer>(m_batch_per_cut_node_sample_count, distribution_slot_count);

		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_prior_pdfs, new_tree_cut_size);
		GenericSoAHelpers::resize<DataContainer>(m_tree_cut_sampling_prior_cdfs, new_tree_cut_size);
	}

	bool free()
	{
		if (maximum_size() == 0)
			return false;

		m_training_records		= DataContainer<IlluminationAwareKDTreeNEEDistributionTrainingRecord>();
		m_training_record_count = DataContainer<GenericAtomicType<unsigned int, DataContainer>>();

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
		return m_training_records.size();
	}

	void to_device(IlluminationAwareKDTreeDevice& device)
	{
		device.nee_learnt_distributions.nee_training_records		 = GenericSoAHelpers::get_buffer_data_ptr(m_training_records);
		device.nee_learnt_distributions.nee_training_record_capacity = static_cast<unsigned int>(m_training_records.size());
		device.nee_learnt_distributions.nee_training_record_count	 = GenericSoAHelpers::get_buffer_data_atomic_ptr(m_training_record_count);

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
	}

	DataContainer<IlluminationAwareKDTreeNEEDistributionTrainingRecord> m_training_records;
	DataContainer<GenericAtomicType<unsigned int, DataContainer>> m_training_record_count;

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
