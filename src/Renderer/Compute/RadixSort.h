/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_RADIX_SORT_H
#define RENDERER_COMPUTE_RADIX_SORT_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"

#include <memory>

class RadixSort
{
public:
	RadixSort();
	RadixSort(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	template <typename T>
	void upload_data(const std::vector<T>& keys, const std::vector<T>& values);
	void sort();
	
	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

private:
	void initialize_kernels();

	OrochiBuffer<unsigned int> m_keys_buffer;
	OrochiBuffer<unsigned int> m_values_buffer;
	OrochiBuffer<unsigned int> m_temp_keys_buffer;
	OrochiBuffer<unsigned int> m_temp_values_buffer;
	OrochiBuffer<unsigned int> m_count_table_buffer;

	GPUKernel m_count_kernel;
	GPUKernel m_scan_kernel;
	GPUKernel m_reorder_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;
	
	size_t m_size;
	static constexpr int RADIX_BITS = 8;
	static constexpr int RADIX_SIZE = 1 << RADIX_BITS; // 256
	static constexpr int NUM_PASSES = 32 / RADIX_BITS; // 4 passes for 32-bit keys
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
