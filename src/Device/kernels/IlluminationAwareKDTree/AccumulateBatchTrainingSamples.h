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
inline accumulate_batch_training_samples(IlluminationAwareKDTreeDevice illumination_aware_kd_tree, int x)
#else
GLOBAL_KERNEL_SIGNATURE(void) accumulate_batch_training_samples(IlluminationAwareKDTreeDevice illumination_aware_kd_tree)
#endif
{
#ifdef __KERNELCC__
	const uint32_t sample_index = blockIdx.x * blockDim.x + threadIdx.x;
#else
	const uint32_t sample_index = x;
#endif
	const uint32_t sample_count = *illumination_aware_kd_tree.training_sample_count;

	// Threads beyond the current compact sample array do nothing.
	if (sample_index >= sample_count)
		return;

	illumination_aware_kd_tree.accumulate_sample_into_existing_tree(illumination_aware_kd_tree.training_samples[sample_index]);
}

#endif
