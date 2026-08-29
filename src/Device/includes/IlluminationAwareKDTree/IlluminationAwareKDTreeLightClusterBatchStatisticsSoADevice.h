/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_SOA_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_SOA_DEVICE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLightClusterBatchStatistics.h"
#include "HostDeviceCommon/AtomicType.h"

struct IlluminationAwareKDTreeLightClusterBatchStatisticsSoADevice
{
	// Conversion helpers keep the existing batch-update code readable while storing each
	// independently accumulated value in an atomically addressable scalar array.
	HIPRT_DEVICE IlluminationAwareKDTreeLightClusterBatchStatistics read(unsigned int index) const
	{
		IlluminationAwareKDTreeLightClusterBatchStatistics statistics{};

		statistics.contribution_sum			= contribution_sum[index];
		statistics.squared_contribution_sum = squared_contribution_sum[index];
		statistics.selected_count			= selected_count[index];

		return statistics;
	}

	HIPRT_DEVICE void write(unsigned int index, const IlluminationAwareKDTreeLightClusterBatchStatistics& statistics)
	{
		contribution_sum[index]			= statistics.contribution_sum;
		squared_contribution_sum[index] = statistics.squared_contribution_sum;
		selected_count[index]			= statistics.selected_count;
	}

	HIPRT_DEVICE void reset(unsigned int index)
	{
		contribution_sum[index]			= 0.0f;
		squared_contribution_sum[index] = 0.0f;
		selected_count[index]			= 0u;
	}

	AtomicType<float>* contribution_sum			= nullptr;
	AtomicType<float>* squared_contribution_sum = nullptr;
	AtomicType<unsigned int>* selected_count	= nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_LIGHT_CLUSTER_BATCH_STATISTICS_SOA_DEVICE_H
