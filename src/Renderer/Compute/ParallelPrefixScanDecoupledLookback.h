/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_PARALLEL_PREFIX_SCAN_ONE_PASS_H
#define RENDERER_COMPUTE_PARALLEL_PREFIX_SCAN_ONE_PASS_H

#include "Compiler/GPUKernel.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"
#include "HIPRT-Orochi/OrochiBuffer.h"

/**
 * Reference: Fast CDF generation on the GPU for light picking
 * https://blog.traverseresearch.nl/fast-cdf-generation-on-the-gpu-for-light-picking-5c50b97c552b
 */
class ParallelPrefixScanDecoupledLookback
{
public:
	ParallelPrefixScanDecoupledLookback();
	ParallelPrefixScanDecoupledLookback(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	bool is_setup();
	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void initialize_kernels();

	void upload_data(const std::vector<unsigned int>& data);
	void scan();

	OrochiBuffer<unsigned int>& get_output_buffer();

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

private:
	OrochiBuffer<unsigned int> m_input_buffer;
	OrochiBuffer<unsigned int> m_output_buffer;
	OrochiBuffer<unsigned int> m_global_block_index_counter_buffer;
	OrochiBuffer<ParallelPrefixScanDecoupledLookbackBlockDescriptor> m_block_descriptors_buffer;

	GPUKernel m_block_scan_kernel;
	GPUKernel m_block_descriptor_init_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	unsigned int m_size;
};

#endif
