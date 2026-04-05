/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_H
#define RENDERER_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/Compute/DataTransforms/ComputeDataTransforms.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"

#include <functional>

template <typename InputType, typename TransformedType = InputType, typename OutputType = InputType>
class ParallelSegmentedReduction
{
public:
	ParallelSegmentedReduction();
	ParallelSegmentedReduction(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	void init(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void set_input_id_transform(std::unique_ptr<ComputeInputIDTransform> transform);
	void set_data_transform(std::unique_ptr<ComputeDataTransform> transform);
	void compile();

	void resize(unsigned int element_count);
	void free();

	/**
	 * @param data Input data to scan
	 * @param flags Segment flags. The prefix scans will be computed separately for each segment. This vector should contain packed bits, i.e. one unsigned int
	 * contains 32 flags. This vector should therefore be of size ceil(data.size() / 32)
	 */
	void upload_input_data(const std::vector<InputType>& data, const std::vector<unsigned int>& flags);
	void set_data_pointers(InputType* input_buffer_pointer, unsigned int element_count);
	void set_data_pointers(InputType* input_buffer_pointer, unsigned int* flags_buffer_pointer, unsigned int element_count);

	unsigned int get_evenly_spaced_segment_size();
	void set_evenly_spaced_segment_size(unsigned int segment_size);

	void reduce(bool auto_stream_synchronize = true);

	float get_last_execution_time();

	constexpr std::string get_input_data_type_as_string() const;
	constexpr std::string get_transformed_data_type_as_string() const;
	constexpr std::string get_output_data_type_as_string() const;

	OrochiBuffer<OutputType>& get_output_buffer();
	OrochiBuffer<unsigned int>& get_segment_ids_buffer();

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type_uint_to_float(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_evenly_spaced_flags(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	template <typename InputDataType, typename TransformedDataType = InputDataType, typename OutputDataType = InputDataType>
	static void unit_test_template(
							std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
							oroStream_t stream,
							ParallelSegmentedReduction<InputDataType, TransformedDataType, OutputDataType>& reducer,
							std::function<TransformedDataType(InputDataType&)> input_value_transform   = [](InputDataType val) { return val; },
							std::function<OutputDataType(TransformedDataType&)> output_value_transform = [](TransformedDataType val) { return val; });

private:
	void initialize_kernels();

private:
	OrochiBuffer<InputType> m_input_buffer;
	OrochiBuffer<OutputType> m_output_buffer;
	OrochiBuffer<unsigned int> m_flags_buffer;

	InputType* m_input_data_pointer			  = nullptr;
	unsigned int* m_flags_data_pointer		  = nullptr;
	unsigned int m_evenly_spaced_segment_size = 0;

	GPUKernel m_scan_kernel;
	ParallelPrefixScanDecoupledLookback<unsigned int> m_segment_ids_prefix_scan;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	unsigned int m_size;
	unsigned int m_last_resize_element_count = 0;
};

#include "Renderer/Compute/ParallelSegmentedReduction.inl"

#endif
