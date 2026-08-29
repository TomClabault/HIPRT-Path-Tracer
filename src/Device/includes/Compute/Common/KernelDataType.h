/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_KERNEL_DATA_TYPE_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_KERNEL_DATA_TYPE_H

#ifndef __KERNELCC__
using InputDataType		  = unsigned int;
using TransformedDataType = unsigned int;
using OutputDataType	  = unsigned int;
#endif // #ifndef __KERNELCC__

#endif // #ifndef DEVICE_INCLUDES_COMPUTE_COMMON_KERNEL_DATA_TYPE_H
