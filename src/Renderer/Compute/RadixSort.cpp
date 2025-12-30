/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Compute/RadixSort.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

RadixSort::RadixSort() : m_hiprt_ctx(nullptr), m_stream(nullptr), m_size(0)
{
	initialize_kernels();
}

RadixSort::RadixSort(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream) 
	: m_hiprt_ctx(hiprt_ctx), m_stream(stream), m_size(0)
{
	initialize_kernels();
}

void RadixSort::initialize_kernels()
{
	m_count_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Count.h");
	m_count_kernel.set_kernel_function_name("RadixSort_Count");
	
	m_scan_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Scan.h");
	m_scan_kernel.set_kernel_function_name("RadixSort_Scan");
	
	m_reorder_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Reorder.h");
	m_reorder_kernel.set_kernel_function_name("RadixSort_Reorder");
}

void RadixSort::set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	m_hiprt_ctx = hiprt_ctx;
	m_stream = stream;
}

void RadixSort::sort()
{
	if (m_size == 0 || !m_hiprt_ctx || !m_stream)
		return;

	// Compile kernels if not already compiled
	if (!m_count_kernel.has_been_compiled())
		m_count_kernel.compile(m_hiprt_ctx);
	if (!m_scan_kernel.has_been_compiled())
		m_scan_kernel.compile(m_hiprt_ctx);
	if (!m_reorder_kernel.has_been_compiled())
		m_reorder_kernel.compile(m_hiprt_ctx);

	// Allocate temporary buffers if needed
	if (m_temp_keys_buffer.size() < m_size)
		m_temp_keys_buffer.resize(m_size);
	if (m_temp_values_buffer.size() < m_size)
		m_temp_values_buffer.resize(m_size);
	if (m_count_table_buffer.size() < RADIX_SIZE)
		m_count_table_buffer.resize(RADIX_SIZE);

	// Determine number of thread blocks (256 threads per block is common for radix sort)
	constexpr int THREADS_PER_BLOCK = 256;
	int num_blocks = (m_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	unsigned int* input_keys = m_keys_buffer.get_device_pointer();
	unsigned int* input_values = m_values_buffer.get_device_pointer();
	unsigned int* output_keys = m_temp_keys_buffer.get_device_pointer();
	unsigned int* output_values = m_temp_values_buffer.get_device_pointer();
	unsigned int* count_table = m_count_table_buffer.get_device_pointer();

	// Perform radix sort in multiple passes
	for (int pass = 0; pass < NUM_PASSES; pass++)
	{
		int bit_offset = pass * RADIX_BITS;

		// Zero the count table before counting
		m_count_table_buffer.memset_whole_buffer(0);

		// Count kernel: Count occurrences of each radix digit
		void* count_args[] = { &input_keys, &count_table, &m_size, &bit_offset };
		m_count_kernel.launch_asynchronous(THREADS_PER_BLOCK, 1, m_size, 1, count_args, m_stream);

		// Scan kernel: Compute exclusive prefix sum
		void* scan_args[] = { &count_table };
		m_scan_kernel.launch_asynchronous(THREADS_PER_BLOCK, 1, RADIX_SIZE, 1, scan_args, m_stream);

		// Reorder kernel: Scatter elements to sorted positions
		void* reorder_args[] = { &input_keys, &input_values, &output_keys, &output_values, 
								 &count_table, &m_size, &bit_offset };
		m_reorder_kernel.launch_asynchronous(THREADS_PER_BLOCK, 1, m_size, 1, reorder_args, m_stream);

		// Synchronize stream to ensure all kernels complete before next pass
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_stream));

		// Swap buffers for next pass
		if (pass < NUM_PASSES - 1)
		{
			// Swap the buffer objects, not just pointers
			std::swap(m_keys_buffer, m_temp_keys_buffer);
			std::swap(m_values_buffer, m_temp_values_buffer);
			
			// Update pointers for next iteration
			input_keys = m_keys_buffer.get_device_pointer();
			input_values = m_values_buffer.get_device_pointer();
			output_keys = m_temp_keys_buffer.get_device_pointer();
			output_values = m_temp_values_buffer.get_device_pointer();
		}
	}

	// If we had an odd number of passes, the final result is in temp buffers, so copy back
	if (NUM_PASSES % 2 == 1)
	{
		m_keys_buffer.memcpy_from(m_temp_keys_buffer);
		m_values_buffer.memcpy_from(m_temp_values_buffer);
	}
}
