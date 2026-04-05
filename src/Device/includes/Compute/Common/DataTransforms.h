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

#define INPUT_ID_TRANSFORM(input_id, input_size)   return input_id;
#define INPUT_TRANSFORM(value, global_element_id)  return value;
#define OUTPUT_TRANSFORM(value, global_element_id) return value;

#define INPUT_ID_TRANSFORM_STRING STR(INPUT_ID_TRANSFORM(x, y))
#define INPUT_TRANSFORM_STRING	  STR(INPUT_TRANSFORM(x, y))
#define OUTPUT_TRANSFORM_STRING	  STR(OUTPUT_TRANSFORM(x, y))
#endif

namespace ComputeDataTransforms
{
	HIPRT_DEVICE HIPRT_INLINE unsigned int input_id_transform(unsigned int input_id, unsigned int input_size)
	{
		INPUT_ID_TRANSFORM(input_id, input_size);
	}

	HIPRT_DEVICE HIPRT_INLINE DataType input_value_transform(DataType value, unsigned int global_element_id)
	{
		INPUT_TRANSFORM(value, global_element_id);
	}

	HIPRT_DEVICE HIPRT_INLINE DataType output_value_transform(DataType value, unsigned int global_element_id)
	{
		OUTPUT_TRANSFORM(value, global_element_id);
	}
} // namespace ComputeDataTransforms

#endif
