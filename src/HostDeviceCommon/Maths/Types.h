/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */                                                                                                                                                            \


#ifndef HOST_DEVICE_COMMON_MATHS_TYPES_H
#define HOST_DEVICE_COMMON_MATHS_TYPES_H

#ifdef __KERNELCC__
#ifdef __CUDACC__
#include "cuda_fp16.h"

using fp16 = half;
#else
using fp16 = __half;
#endif
#else // !__KERNELCC__

using fp16 = float;

#endif // __KERNELCC__

#endif
