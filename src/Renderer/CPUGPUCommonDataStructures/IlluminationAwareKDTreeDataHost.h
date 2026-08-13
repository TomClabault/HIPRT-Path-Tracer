/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeCoreDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeNEELearntDistributionsDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeNISMLDataHost.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	static constexpr unsigned int MAXIMUM_NUMBER_OF_NODES = IlluminationAwareKDTreeCoreDataHost<DataContainer>::MAXIMUM_NUMBER_OF_NODES;

	void resize(unsigned int new_node_capacity, unsigned int new_training_sample_capacity, int new_tree_cut_size)
	{
		m_kd_tree_data.resize(new_node_capacity, new_training_sample_capacity);
		m_nisml_data.resize(new_node_capacity);
		m_nee_learnt_distributions_data.resize(new_node_capacity, new_training_sample_capacity, new_tree_cut_size);
	}

	void reset()
	{
		// All reset is already done by the render pass
	}

	bool free()
	{
		bool core_data_freed					 = m_kd_tree_data.free();
		bool nisml_data_freed					 = m_nisml_data.free();
		bool nee_learnt_distributions_data_freed = m_nee_learnt_distributions_data.free();

		return core_data_freed || nisml_data_freed || nee_learnt_distributions_data_freed;
	}

	std::size_t maximum_size() const
	{
		return m_kd_tree_data.maximum_size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice device = m_kd_tree_data.to_device();

		m_nisml_data.to_device(device);
		m_nee_learnt_distributions_data.to_device(device);

		device.user_settings								  = render_data.illumination_aware_kd_tree.user_settings;
		device.nee_learnt_distributions.learning_nee_settings = render_data.illumination_aware_kd_tree.nee_learnt_distributions.learning_nee_settings;

		return device;
	}

	IlluminationAwareKDTreeCoreDataHost<DataContainer> m_kd_tree_data;
	IlluminationAwareKDTreeNISMLDataHost<DataContainer> m_nisml_data;
	IlluminationAwareKDTreeNEELearntDistributionsDataHost<DataContainer> m_nee_learnt_distributions_data;
};

#endif
