/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_UTILITY_MEMSET_KERNEL_H
#define DEVICE_KERNELS_UTILITY_MEMSET_KERNEL_H

#include "Device/includes/FixIntellisense.h"

#ifndef __KERNELCC__
// Just so the compiler and intellisense are happy
#define DATA_TYPE float
#define VALUE 0.0f
#endif

GLOBAL_KERNEL_SIGNATURE(void) Memset(DATA_TYPE* buffer, unsigned int size)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	buffer[idx] = static_cast<DATA_TYPE>(VALUE);
}

#endif
