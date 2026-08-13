/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_BATCH_TRAINING_SAMPLES_H
#define DEVICE_KERNELS_ILLUMINATION_AWARE_KD_TREE_ACCUMULATE_BATCH_TRAINING_SAMPLES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"

#ifndef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline IlluminationAwareKDTree_AccumulateBatchTrainingSamples(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void) IlluminationAwareKDTree_AccumulateBatchTrainingSamples(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif
{
#ifdef __KERNELCC__
	unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int sample_index = x;
#endif

	unsigned int sample_count = *illumination_aware_kd_tree.core.training_sample_count;
	if (sample_index >= sample_count)
		return;

	illumination_aware_kd_tree.core.accumulate_sample_into_existing_tree(illumination_aware_kd_tree.core.training_samples[sample_index]);
}

#endif
