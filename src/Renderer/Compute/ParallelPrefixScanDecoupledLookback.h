/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
template <typename T>
class ParallelPrefixScanDecoupledLookback
{
public:
	ParallelPrefixScanDecoupledLookback();
	ParallelPrefixScanDecoupledLookback(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	bool is_setup();
	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void initialize_kernels();
	void compile();

	void resize(unsigned int element_count);
	void upload_input_data(const std::vector<T>& data);
	void set_data_pointers(T* input_buffer_pointer, unsigned int element_count);
	void set_data_pointers(T* input_buffer_pointer, T* output_buffer_pointer, unsigned int element_count);
	void set_transform(std::unique_ptr<ComputeDataTransform> transform);
	void scan(bool auto_stream_synchronize = true);

	float get_last_execution_time();

	constexpr std::string get_data_type_as_string();

	OrochiBuffer<T>& get_output_buffer();

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

private:
	static void unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	static void unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	template <typename DataType>
	static void unit_test_template(
							std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
							oroStream_t stream,
							ParallelPrefixScanDecoupledLookback<DataType>& scanner,
							std::function<DataType(DataType&)> input_value_transform  = [](DataType val) { return val; },
							std::function<DataType(DataType&)> output_value_transform = [](DataType val) { return val; });

private:
	OrochiBuffer<T> m_input_buffer;
	OrochiBuffer<T> m_output_buffer;

	T* m_input_data_pointer	 = nullptr;
	T* m_output_data_pointer = nullptr;

	OrochiBuffer<unsigned int> m_global_block_index_counter_buffer;
	OrochiBuffer<ParallelPrefixScanDecoupledLookbackBlockDescriptor> m_block_descriptors_buffer;

	GPUKernel m_scan_kernel;
	GPUKernel m_block_descriptor_init_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	unsigned int m_size;
	unsigned int m_last_resize_element_count = 0;
};

#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.inl"

#endif
