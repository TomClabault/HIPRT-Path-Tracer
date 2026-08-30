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
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLearningToClusterDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeNISMLDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeSpatialSampleMomentsSoAHost.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeDataHost
{
	void resize(unsigned int new_node_capacity,
				unsigned int new_training_sample_capacity,
				unsigned int new_nisml_representative_capacity		   = 1,
				unsigned int new_nisml_hash_table_reserved_bytes	   = 100000000u,
				unsigned int new_nisml_hash_normal_precision		   = 2u,
				unsigned int maximum_learning_to_cluster_lightcut_size = LearningToClusterMaximumLightCutSize)
	{
		m_kd_tree_data.resize(new_node_capacity, new_training_sample_capacity);
		m_nisml_data.resize(new_node_capacity, new_nisml_representative_capacity, new_nisml_hash_table_reserved_bytes, new_nisml_hash_normal_precision);
		m_learning_to_cluster_data.resize(new_node_capacity, new_training_sample_capacity, maximum_learning_to_cluster_lightcut_size);

		GenericSoAHelpers::resize<DataContainer>(m_any_cell_needs_split, 1);
		GenericSoAHelpers::resize_host_pinned_mem(m_any_cell_needs_split_host_pinned, 1);
	}

	void reset()
	{
		// All reset is already done by the render pass
	}

	bool free()
	{
		if (m_any_cell_needs_split.size() == 0)
			return false;

		bool core_data_freed				= m_kd_tree_data.free();
		bool nisml_data_freed				= m_nisml_data.free();
		bool learning_to_cluster_data_freed = m_learning_to_cluster_data.free();

		bool any_cell_needs_split_freed = m_any_cell_needs_split.size() > 0;
		m_any_cell_needs_split			= DataContainer<unsigned char>();

		bool any_cell_needs_split_host_pinned_freed = m_any_cell_needs_split_host_pinned.size() > 0;
		m_any_cell_needs_split_host_pinned			= DataContainer<unsigned char>();

		return core_data_freed || nisml_data_freed || learning_to_cluster_data_freed || any_cell_needs_split_freed || any_cell_needs_split_host_pinned_freed;
	}

	std::size_t maximum_size() const
	{
		return m_kd_tree_data.maximum_size();
	}

	IlluminationAwareKDTreeDevice to_device(HIPRTRenderData& render_data)
	{
		IlluminationAwareKDTreeDevice kd_tree_device = m_kd_tree_data.to_device();

		m_nisml_data.to_device(kd_tree_device);
		m_learning_to_cluster_data.to_device(kd_tree_device);
		kd_tree_device.any_cell_needs_split = GenericSoAHelpers::get_buffer_data_ptr(m_any_cell_needs_split);

		kd_tree_device.core.user_settings								   = render_data.kd_tree_device.core.user_settings;
		kd_tree_device.learning_to_cluster.user_settings				   = render_data.kd_tree_device.learning_to_cluster.user_settings;
		kd_tree_device.learning_to_cluster.effective_initial_lightcut_size = render_data.kd_tree_device.learning_to_cluster.effective_initial_lightcut_size;

		return kd_tree_device;
	}

	IlluminationAwareKDTreeCoreDataHost<DataContainer> m_kd_tree_data;
	IlluminationAwareKDTreeNISMLDataHost<DataContainer> m_nisml_data;
	IlluminationAwareKDTreeLearningToClusterDataHost<DataContainer> m_learning_to_cluster_data;

	DataContainer<unsigned char> m_any_cell_needs_split;
	DataContainer<unsigned char> m_any_cell_needs_split_host_pinned;
};

#endif // #ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_DATA_HOST_H
