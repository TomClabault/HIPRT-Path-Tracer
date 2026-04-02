/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_COMPUTE_RADIX_SORT_H
#define RENDERER_COMPUTE_RADIX_SORT_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"
#include "Renderer/Compute/ParallelSegmentedPrefixScan.h"

#include <memory>

class RadixSort
{
public:
	enum class Ordering
	{
		ASCENDING,
		DESCENDING
	};

	RadixSort();
	RadixSort(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	void set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

	/**
	 *
	 */
	void resize(unsigned int element_count);

	void upload_input_data(const std::vector<unsigned int>& keys, const std::vector<unsigned int>& values);
	void set_data_pointers(unsigned int* keys_device_pointer, unsigned int* values_device_pointer, unsigned int element_count);
	void sort();

	OrochiBuffer<unsigned int>& get_sorted_keys_buffer();
	OrochiBuffer<unsigned int>& get_sorted_values_buffer();

	void set_ordering(Ordering order);

	static void unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream);

private:
	void initialize_kernels();

	OrochiBuffer<unsigned int> m_keys_buffer;
	OrochiBuffer<unsigned int> m_values_buffer;
	unsigned int* m_keys_data_pointer	= nullptr;
	unsigned int* m_values_data_pointer = nullptr;

	OrochiBuffer<unsigned int> m_temp_keys_buffer;
	OrochiBuffer<unsigned int> m_temp_values_buffer;
	OrochiBuffer<unsigned int> m_global_count_tables_buffer;
	OrochiBuffer<unsigned int> m_per_block_count_tables_buffer;
	OrochiBuffer<unsigned int> m_per_block_count_tables_scanned_buffer;

	GPUKernel m_memset_0_kernel;
	GPUKernel m_count_kernel;
	ParallelPrefixScanDecoupledLookback<unsigned int> m_global_count_table_prefix_scan;
	ParallelSegmentedPrefixScan m_per_block_count_table_prefix_scan;
	GPUKernel m_reorder_kernel;

	std::shared_ptr<HIPRTOrochiCtx> m_hiprt_ctx;
	oroStream_t m_stream;

	size_t m_size;
	unsigned int m_last_resize_element_count = 0;
	bool m_data_uploaded					 = false;

	Ordering m_ordering = Ordering::ASCENDING;
};

#endif
