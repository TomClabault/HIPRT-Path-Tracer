/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_PARALLEL_PREFIX_SCAN_H
#define RENDERER_COMPUTE_PARALLEL_PREFIX_SCAN_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"

/**
 * Reference: [GPU Gems 3, Chapter 39. Parallel Prefix Sum (Scan) with CUDA]
 * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 */
class ParallelPrefixScan
{
public:
	ParallelPrefixScan();
	ParallelPrefixScan(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	bool is_setup();
	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);
	void initialize_kernels();

	void upload_data(const std::vector<unsigned int>& data);
	void scan();

	OrochiBuffer<unsigned int>& get_output_buffer();

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

private:
	OrochiBuffer<unsigned int> m_input_buffer;
	
	/**
	 * If the input data is larger than a block size, we need to store the sums of each
	 * chunk of the input and these sums are going to be used to reconstruct the full output
	 * 
	 * The m_level_0_block_sums buffer stores the sums of each chunk of size PARALLEL_PREFIX_SCAN_CHUNK_SIZE
	 * and is itself going to be scanned. If this buffer is larger than a block size as well,
	 * then we're going to need to store its sums as well, in m_level_1_block_sums, and so on...
	 * 
	 * 3 levels are supported at most which is enough for 2^32 elements input (about 4 billion elements)
	 */
	OrochiBuffer<unsigned int> m_level_0_block_sums;
	OrochiBuffer<unsigned int> m_scanned_level_0_blocks_sums;
	OrochiBuffer<unsigned int> m_level_1_block_sums;
	OrochiBuffer<unsigned int> m_scanned_level_1_blocks_sums;
	OrochiBuffer<unsigned int> m_level_2_block_sums;
	OrochiBuffer<unsigned int> m_scanned_level_2_blocks_sums;
	unsigned int m_hierarchy_levels_used = 0;

	OrochiBuffer<unsigned int> m_output_buffer;

	GPUKernel m_block_scan_kernel;
	GPUKernel m_block_increment_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	unsigned int m_size_padded;
	unsigned int m_size_non_padded;
};

#endif
