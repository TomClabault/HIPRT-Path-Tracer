/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
template <typename T>
class ParallelSegmentedPrefixScan
{
public:
	ParallelSegmentedPrefixScan();
	ParallelSegmentedPrefixScan(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void set_transform(std::unique_ptr<ComputeDataTransform> transform);
	void compile();

	void resize(unsigned int element_count);
	/**
	 * @param data Input data to scan
	 * @param flags Segment flags. The prefix scans will be computed separately for each segment. This vector should contain packed bits, i.e. one unsigned int
	 * contains 32 flags. This vector should therefore be of size ceil(data.size() / 32)
	 */
	void upload_input_data(const std::vector<T>& data, const std::vector<unsigned int>& flags);
	void set_data_pointers(T* input_buffer_pointer, unsigned int element_count);
	void set_data_pointers(T* input_buffer_pointer, unsigned int* flags_buffer_pointer, unsigned int element_count);
	void set_evenly_spaced_segment_size(unsigned int segment_size);
	void scan(bool auto_stream_synchronize = true);

	float get_last_execution_time();

	constexpr std::string get_data_type_as_string() const;

	OrochiBuffer<T>& get_output_buffer();

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	template <typename DataType>
	static void unit_test_template(
							std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
							oroStream_t stream,
							ParallelSegmentedPrefixScan<DataType>& scanner,
							std::function<DataType(DataType&)> input_value_transform  = [](DataType val) { return val; },
							std::function<DataType(DataType&)> output_value_transform = [](DataType val) { return val; });

private:
	void initialize_kernels();

private:
	OrochiBuffer<T> m_input_buffer;
	OrochiBuffer<unsigned int> m_flags_buffer;

	T* m_input_data_pointer					  = nullptr;
	unsigned int* m_flags_data_pointer		  = nullptr;
	unsigned int m_evenly_spaced_segment_size = 0;

	OrochiBuffer<T> m_output_buffer;

	OrochiBuffer<unsigned int> m_global_block_index_counter_buffer;
	OrochiBuffer<ParallelPrefixScanDecoupledLookbackBlockDescriptor> m_block_descriptors_buffer;

	GPUKernel m_scan_kernel;
	GPUKernel m_block_descriptor_init_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	unsigned int m_size;
	unsigned int m_last_resize_element_count = 0;
};

#include "Renderer/Compute/ParallelSegmentedPrefixScan.inl"

#endif
