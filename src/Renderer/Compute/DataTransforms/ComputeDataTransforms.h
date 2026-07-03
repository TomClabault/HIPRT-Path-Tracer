/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_DATA_TRANSFORM_COMPUTE_DATA_TRANSFORMS_H
#define RENDERER_COMPUTE_DATA_TRANSFORM_COMPUTE_DATA_TRANSFORMS_H

#include "Device/includes/Compute/Common/DataTransforms.h"

#include <string_view>

/**
 * Allows a compute kernel to modify the element read from memory before processing it. For an input data transform, this allows for example running a parallel
 * prefix scan on unsigned int that actually contain packed data and we want the parallel prefix sum to prefix-sum these elements packed into the unsigned ints.
 * We would thus have to write a data transform that takes an unsigned int as input extracts the packed data and returns it. The returned value are the actual
 * values that are going to be prefixed sum.
 *
 * An output data transform allows to modify the value before writing it back to memory. Simple.
 *
 * An input ID data transform allows modifying the element that each thread is going to read from. Without an ID transform, compute kernels usually read the
 * element at the index corresponding to their global thread ID. With an input ID transform, the kernel can modify this index, for example to have threads read
 * from a different order or to have 32 threads in a row read individual bits of an unsigned int values (if we want to prefix sum a list of flags where each
 * flag is a bit but packed into an unsigned int for example)
 *
 * The signature of those transforms are given in DataTransforms.h
 */
class ComputeDataTransform
{
public:
	static inline std::string INPUT_DATA_TRANSFORM_STRING_STUB	= INPUT_DATA_TRANSFORM_STRING;
	static inline std::string OUTPUT_DATA_TRANSFORM_STRING_STUB = OUTPUT_DATA_TRANSFORM_STRING;

	/**
	 * Input data transform function signature:
	 *
	 * @param value The value read from memory that we want to transform before processing it in the kernel
	 * @param global_element_id The global element ID corresponding to the value read.
	 *
	 * TransformedDataType input_transform(InputDataType value, unsigned int global_element_id)
	 * {
	 *		// Transform 'value' here...
	 *		// ...
	 *
	 *		return value;
	 * }
	 */
	virtual std::string emit_input_transform() const = 0;

	/**
	 * Similar signature as input transform but for output values and OutputDataType everywhere
	 *
	 * OutputDataType output_transform(TransformedDataType value, unsigned int global_element_id)
	 * {
	 *		// Transform 'value' here...
	 *		// ...
	 *
	 *		return value;
	 * }
	 */
	virtual std::string emit_output_transform() const = 0;
};

class ComputeDataTransformIdentity : public ComputeDataTransform
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

class ComputeDataTransformMultiplyBy2 : public ComputeDataTransform
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

class ComputeDataTransformUnpackUintBits : public ComputeDataTransform
{
public:
	virtual std::string emit_input_transform() const override
	{
		// Each unsigned int contains 32 bits, so we want to have 32 threads in a row read the bits of the same unsigned int. We can achieve this by having the
		// input transform return the bit corresponding to the thread's global thread ID modulo 32.
		return "return (value >> (global_element_id & 31)) & 1;";
	}

	virtual std::string emit_output_transform() const override
	{
		return "return value;";
	}
};

class ComputeInputIDTransform
{
public:
	static inline std::string INPUT_ID_TRANSFORM_STRING_STUB = INPUT_ID_TRANSFORM_STRING;

	/**
	 * Signature of the input ID transform function:
	 *
	 * unsigned int input_id_transform(unsigned int input_id, unsigned int input_size)
	 * {
	 *		// Transform 'input_id' here. 'input_id' is the index of the element
	 *		// that the thread would read without an ID transform, usually corresponding to
	 *		// the global thread ID.
	 *
	 *		// ...
	 *
	 *		return input_id;
	 */
	virtual std::string emit_input_id_transform() const = 0;
};

class ComputeInputIDTransformIdentity : public ComputeInputIDTransform
{
public:
	virtual std::string emit_input_id_transform() const override
	{
		return "return input_id;";
	}
};

class ComputeInputIDTransformUnpackUintBits : public ComputeInputIDTransform
{
public:
	virtual std::string emit_input_id_transform() const override
	{
		// Each unsigned int contains 32 bits, so we want to have 32 threads in a row read the bits of the same unsigned int. We can achieve this by having the
		// input ID transform return the global thread ID divided by 32.
		return "return input_id / 32;";
	}
};

#endif
