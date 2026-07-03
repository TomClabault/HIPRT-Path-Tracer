/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_PARALLEL_PREFIX_SCAN_ONE_PASS_H
#define RENDERER_COMPUTE_PARALLEL_PREFIX_SCAN_ONE_PASS_H

#include "Compiler/GPUKernel.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/Compute/DataTransforms/ComputeDataTransforms.h"

#include <functional>

/**
 * Reference: Single-pass Parallel Prefix Scan with Decoupled Look-back
 * https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 */
template <typename InputType, typename TransformedType = InputType, typename OutputType = InputType>
class ParallelPrefixScanDecoupledLookback
{
public:
	ParallelPrefixScanDecoupledLookback();
	ParallelPrefixScanDecoupledLookback(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	void init(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void set_input_id_transform(std::unique_ptr<ComputeInputIDTransform> id_transform);
	void set_data_transform(std::unique_ptr<ComputeDataTransform> transform);
	void set_exclusive_scan(bool exclusive);
	void compile();

	void resize(unsigned int element_count);
	void free();

	void upload_input_data(const std::vector<InputType>& data);
	void set_data_pointers(InputType* input_buffer_pointer, unsigned int element_count);
	void set_data_pointers(InputType* input_buffer_pointer, OutputType* output_buffer_pointer, unsigned int element_count);
	void scan(bool auto_stream_synchronize = true);

	float get_last_execution_time();

	constexpr std::string get_input_data_type_as_string() const;
	constexpr std::string get_transformed_data_type_as_string() const;
	constexpr std::string get_output_data_type_as_string() const;

	OrochiBuffer<OutputType>& get_output_buffer();

	std::size_t get_byte_size() const;

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

private:
	void initialize_kernels();

private:
	static void unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type_uint_to_float(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_inclusive(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	template <typename InputDataType, typename TransformedDataType = InputDataType, typename OutputDataType = InputDataType>
	static void unit_test_template(
		std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
		oroStream_t stream,
		ParallelPrefixScanDecoupledLookback<InputDataType, TransformedDataType, OutputDataType>& scanner,
		std::function<TransformedDataType(InputDataType&)> input_value_transform   = [](InputDataType val) { return val; },
		std::function<OutputDataType(TransformedDataType&)> output_value_transform = [](TransformedDataType val) { return val; },
		bool exclusive_scan														   = true);

private:
	OrochiBuffer<InputType> m_input_buffer;
	OrochiBuffer<OutputType> m_output_buffer;

	InputType* m_input_data_pointer	  = nullptr;
	OutputType* m_output_data_pointer = nullptr;

	OrochiBuffer<unsigned int> m_global_block_index_counter_buffer;
	OrochiBuffer<ParallelPrefixScanDecoupledLookbackBlockDescriptor> m_block_descriptors_buffer;

	bool m_exclusive_scan = true;
	GPUKernel m_scan_kernel;
	GPUKernel m_block_descriptor_init_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	unsigned int m_size;
	unsigned int m_last_resize_element_count = 0;
};

#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.inl"

#endif
