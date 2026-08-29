/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_SOA_HOST_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_SOA_HOST_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
struct IlluminationAwareKDTreeLightClusterBatchStatisticsSoAHost
{
	using Data = GenericSoA<DataContainer,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<float, DataContainer>,
							GenericAtomicType<unsigned int, DataContainer>>;

	enum Buffers
	{
		CONTRIBUTION_SUM,
		SQUARED_CONTRIBUTION_SUM,
		SELECTED_COUNT,
	};

	void resize(unsigned int new_size)
	{
		data.resize(new_size);
	}

	void free()
	{
		data.free();
	}

	std::size_t get_byte_size() const
	{
		return data.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return data.maximum_size();
	}

	IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice to_device()
	{
		IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice device;

		device.contribution_sum			= data.template get_buffer_data_atomic_ptr<CONTRIBUTION_SUM>();
		device.squared_contribution_sum = data.template get_buffer_data_atomic_ptr<SQUARED_CONTRIBUTION_SUM>();
		device.selected_count			= data.template get_buffer_data_atomic_ptr<SELECTED_COUNT>();

		return device;
	}

	Data data;
};

#endif // #ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_SOA_HOST_H
