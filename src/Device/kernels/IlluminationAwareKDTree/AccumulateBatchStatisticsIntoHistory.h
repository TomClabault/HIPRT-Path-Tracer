/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_AccumulateBatchStatisticsIntoHistory(IlluminationAwareKDTreeDevice kd_tree_device, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void) IlluminationAwareKDTree_AccumulateBatchStatisticsIntoHistory(IlluminationAwareKDTreeDevice kd_tree_device)
#endif
{
#ifdef __KERNELCC__
	const uint32_t node_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	const uint32_t node_index = x;
#endif

	const uint32_t node_count = *kd_tree_device.core.node_count;

	// Only currently allocated physical nodes are valid.
	if (node_index >= node_count)
		return;

	bool node_allocated = node_index < node_count;
	if (!node_allocated)
		return;

	IlluminationAwareKDTreeIlluminationSignature& history_signature		= kd_tree_device.core.history_signatures[node_index];
	const IlluminationAwareKDTreeIlluminationSignature& batch_signature = kd_tree_device.core.batch_signatures[node_index];

	history_signature.valid_observation_count += batch_signature.valid_observation_count;
	history_signature.scalar_radiance_sum += batch_signature.scalar_radiance_sum;
	history_signature.squared_scalar_radiance_sum += batch_signature.squared_scalar_radiance_sum;
	history_signature.weighted_direction_sum += batch_signature.weighted_direction_sum;

	IlluminationAwareKDTreeSpatialSampleMoments& history_spatial	 = kd_tree_device.core.history_spatial_moments[node_index];
	const IlluminationAwareKDTreeSpatialSampleMoments& batch_spatial = kd_tree_device.core.batch_spatial_moments[node_index];

	history_spatial.positive_radiance_sample_count += batch_spatial.positive_radiance_sample_count;
	history_spatial.position_sum += batch_spatial.position_sum;
	history_spatial.position_squared_sum += batch_spatial.position_squared_sum;
}

#endif
