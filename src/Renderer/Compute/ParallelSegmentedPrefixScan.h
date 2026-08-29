/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_PARALLEL_SEGMENTED_PREFIX_SCAN_ONE_PASS_H
#define RENDERER_COMPUTE_PARALLEL_SEGMENTED_PREFIX_SCAN_ONE_PASS_H

#include "Compiler/GPUKernel.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/Compute/DataTransforms/ComputeDataTransforms.h"

/**
 * Reference: Single-pass Parallel Prefix Scan with Decoupled Look-back
 * https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
 *
 * and
 *
 * Efficient Parallel Scan Algorithms for GPUs
 * https://research.nvidia.com/publication/2008-12_efficient-parallel-scan-algorithms-gpus
 */
template <typename InputType, typename TransformedType = InputType, typename OutputType = InputType>
class ParallelSegmentedPrefixScan
{
public:
	ParallelSegmentedPrefixScan();
	ParallelSegmentedPrefixScan(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	void init(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void set_input_id_transform(std::unique_ptr<ComputeInputIDTransform> id_transform);
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

	bool get_exclusive_scan() const;
	void set_exclusive_scan(bool exclusive);
	void set_evenly_spaced_segment_size(unsigned int segment_size);

	void scan(bool auto_stream_synchronize = true);

	float get_last_execution_time();

	constexpr std::string get_input_data_type_as_string() const;
	constexpr std::string get_transformed_data_type_as_string() const;
	constexpr std::string get_output_data_type_as_string() const;

	OrochiBuffer<OutputType>& get_output_buffer();

	std::size_t get_byte_size() const;

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type_uint_to_float(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_inclusive(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	template <typename InputDataType, typename TransformedDataType = InputDataType, typename OutputDataType = InputDataType>
	static void unit_test_template(
		std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
		oroStream_t stream,
		ParallelSegmentedPrefixScan<InputDataType, TransformedDataType, OutputDataType>& scanner,
		std::function<TransformedDataType(InputDataType&)> input_value_transform   = [](InputDataType val) { return val; },
		std::function<OutputDataType(TransformedDataType&)> output_value_transform = [](TransformedDataType val) { return val; });

private:
	void initialize_kernels();

private:
	OrochiBuffer<InputType> m_input_buffer;
	OrochiBuffer<unsigned int> m_flags_buffer;

	InputType* m_input_data_pointer			  = nullptr;
	unsigned int* m_flags_data_pointer		  = nullptr;
	unsigned int m_evenly_spaced_segment_size = 0;

	OrochiBuffer<OutputType> m_output_buffer;
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

#include "Renderer/Compute/ParallelSegmentedPrefixScan.inl"

#endif // #ifndef RENDERER_COMPUTE_PARALLEL_SEGMENTED_PREFIX_SCAN_ONE_PASS_H
