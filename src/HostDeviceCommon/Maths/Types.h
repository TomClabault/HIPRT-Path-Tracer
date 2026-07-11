/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATHS_TYPES_H
#define HOST_DEVICE_COMMON_MATHS_TYPES_H

#ifdef __KERNELCC__
#ifdef __CUDACC__
#include "cuda_fp16.h"

using fp16 = half;

#else // !__CUDACC__ = __HIPCC__

using fp16 = __half;
typedef _Float16 fp16x16 __attribute__((ext_vector_type(16)));

#endif
#else // !__KERNELCC__

using fp16 = float;

struct fp16x16_struct_
{
	float& operator[](size_t index)
	{
		return x[index];
	}

	float x[16];
};
using fp16x16 = fp16x16_struct_;

#endif // __KERNELCC__

#endif
