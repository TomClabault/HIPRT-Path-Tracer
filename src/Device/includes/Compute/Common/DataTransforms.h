/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_COMMON_DATA_TRANSFORM_H
#define DEVICE_INCLUDES_COMPUTE_COMMON_DATA_TRANSFORM_H

#include "Device/includes/Compute/Common/KernelDataType.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

#ifndef __KERNELCC__
#define STR(x) #x

#define INPUT_ID_TRANSFORM(input_id, input_size)		return input_id;
#define INPUT_DATA_TRANSFORM(value, global_element_id)	return value;
#define OUTPUT_DATA_TRANSFORM(value, global_element_id) return value;

#define INPUT_ID_TRANSFORM_STRING	 STR(INPUT_ID_TRANSFORM(x, y))
#define INPUT_DATA_TRANSFORM_STRING	 STR(INPUT_DATA_TRANSFORM(x, y))
#define OUTPUT_DATA_TRANSFORM_STRING STR(OUTPUT_DATA_TRANSFORM(x, y))
#endif // #ifndef __KERNELCC__

namespace ComputeDataTransforms
{
	HIPRT_DEVICE HIPRT_INLINE unsigned int input_id_transform(unsigned int input_id, unsigned int input_size)
	{
		INPUT_ID_TRANSFORM(input_id, input_size);
	}

	HIPRT_DEVICE HIPRT_INLINE TransformedDataType input_value_transform(InputDataType value, unsigned int global_element_id)
	{
		INPUT_DATA_TRANSFORM(value, global_element_id);
	}

	HIPRT_DEVICE HIPRT_INLINE OutputDataType output_value_transform(TransformedDataType value, unsigned int global_element_id)
	{
		OUTPUT_DATA_TRANSFORM(value, global_element_id);
	}
} // namespace ComputeDataTransforms

#endif // #ifndef DEVICE_INCLUDES_COMPUTE_COMMON_DATA_TRANSFORM_H
