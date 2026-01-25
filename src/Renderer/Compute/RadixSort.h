/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_RADIX_SORT_H
#define RENDERER_COMPUTE_RADIX_SORT_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"

#include <memory>

class RadixSort
{
public:
	RadixSort();
	RadixSort(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	template <typename T>
	void upload_data(const std::vector<T>& keys, const std::vector<T>& values);
	void sort();

	OrochiBuffer<unsigned int>& get_sorted_keys_buffer();
	OrochiBuffer<unsigned int>& get_sorted_values_buffer();

private:
	void initialize_kernels();

	OrochiBuffer<unsigned int> m_keys_buffer;
	OrochiBuffer<unsigned int> m_values_buffer;
	OrochiBuffer<unsigned int> m_temp_keys_buffer;
	OrochiBuffer<unsigned int> m_temp_values_buffer;
	OrochiBuffer<unsigned int> m_count_tables_buffer;

	ParallelPrefixScanDecoupledLookback m_prefix_scan;
	GPUKernel m_count_kernel;
	GPUKernel m_reorder_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;
	
	size_t m_size;
};

template <typename T>
void RadixSort::upload_data(const std::vector<T>& keys, const std::vector<T>& values)
{
	if (keys.size() != values.size())
		return; // Error: sizes don't match

	m_size = keys.size();
	m_keys_buffer.resize(m_size);
	m_values_buffer.resize(m_size);

	m_keys_buffer.upload_data(reinterpret_cast<const unsigned int*>(keys.data()));
	m_values_buffer.upload_data(reinterpret_cast<const unsigned int*>(values.data()));
}

#endif
