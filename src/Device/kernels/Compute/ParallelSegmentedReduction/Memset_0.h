/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_MEMSET_0_H
#define DEVICE_KERNELS_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_MEMSET_0_H

#include "Device/includes/Compute/Common/DataTransforms.h"
#include "Device/includes/FixIntellisense.h"

GLOBAL_KERNEL_SIGNATURE(void)
ParallelSegmentedReduction_Memset_0(OutputDataType* __restrict__ buffer, unsigned int element_count)
{
	unsigned int tid		= threadIdx.x;
	unsigned int bid		= blockIdx.x;
	unsigned int global_tid = bid * blockDim.x + tid;
	if (global_tid < element_count)
		buffer[global_tid] = 0;
}

#endif
