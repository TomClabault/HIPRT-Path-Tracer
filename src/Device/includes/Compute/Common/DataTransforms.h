/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_DATA_TRANSFORM_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_DATA_TRANSFORM_H

#include "Device/includes/Compute/Common/KernelDataType.h"
#include "Device/includes/FixIntellisense.h"

#ifndef __KERNELCC__
#define STR(x) #x

#define INPUT_TRANSFORM(value)	return value;
#define OUTPUT_TRANSFORM(value) return value;

#define INPUT_TRANSFORM_STRING	STR(INPUT_TRANSFORM(x))
#define OUTPUT_TRANSFORM_STRING STR(OUTPUT_TRANSFORM(x))
#endif

HIPRT_DEVICE HIPRT_INLINE DataType input_value_transform(DataType value)
{
	INPUT_TRANSFORM(value);
}

HIPRT_DEVICE HIPRT_INLINE DataType output_value_transform(DataType value)
{
	OUTPUT_TRANSFORM(value);
}

#endif
