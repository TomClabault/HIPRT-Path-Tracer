/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_DATA_TRANSFORM_COMPUTE_DATA_TRANSFORMS_H
#define RENDERER_COMPUTE_DATA_TRANSFORM_COMPUTE_DATA_TRANSFORMS_H

#include "Device/includes/Compute/Common/DataTransforms.h"

#include <string_view>

class ComputeDataTransform
{
public:
	static inline std::string INPUT_TRANSFORM_STRING_STUB  = INPUT_TRANSFORM_STRING;
	static inline std::string OUTPUT_TRANSFORM_STRING_STUB = OUTPUT_TRANSFORM_STRING;

	virtual std::string emit_input_transform() const  = 0;
	virtual std::string emit_output_transform() const = 0;
};

class IdentityTransform : public ComputeDataTransform
{
public:
	virtual std::string emit_input_transform() const override
	{
		return "return value;";
	}

	virtual std::string emit_output_transform() const override
	{
		return "return value;";
	}
};

class MultiplyBy2Transform : public ComputeDataTransform
{
public:
	virtual std::string emit_input_transform() const override
	{
		return "return value * 2;";
	}

	virtual std::string emit_output_transform() const override
	{
		return "return value;";
	}
};

#endif
