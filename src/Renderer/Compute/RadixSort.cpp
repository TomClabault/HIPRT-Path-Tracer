/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/RadixSortCommon.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"
#include "Renderer/Compute/RadixSort.h"

#include <numeric>
#include <random>

RadixSort::RadixSort() : m_hiprt_ctx(nullptr), m_stream(nullptr), m_size(0)
{
}

RadixSort::RadixSort(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream) : m_hiprt_ctx(hiprt_ctx), m_stream(stream), m_size(0)
{
    initialize_kernels();
}

void RadixSort::set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
    m_hiprt_ctx = hiprt_ctx;
    m_stream    = stream;

    initialize_kernels();
}

void RadixSort::initialize_kernels()
{
    m_count_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Count.h");
    m_count_kernel.set_kernel_function_name("RadixSort_Count");
    m_count_kernel.compile(m_hiprt_ctx, {}, true, false);

    m_reorder_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Reorder.h");
    m_reorder_kernel.set_kernel_function_name("RadixSort_Reorder");
    m_reorder_kernel.compile(m_hiprt_ctx, {}, true, false);
}

void RadixSort::sort()
{
    if (m_size == 0 || !m_hiprt_ctx || !m_stream)
        return;

    // Allocate temporary buffers if needed
    if (m_temp_keys_buffer.size() < m_size)
        m_temp_keys_buffer.resize(m_size);
    if (m_temp_values_buffer.size() < m_size)
        m_temp_values_buffer.resize(m_size);
    if (m_count_tables_buffer.size() < RADIX_SORT_RADIX_SIZE)
        m_count_tables_buffer.resize(RADIX_SORT_RADIX_SIZE);

    if (!m_prefix_scan.is_setup())
        m_prefix_scan.set_context(m_hiprt_ctx, m_stream);

    unsigned int* input_keys    = m_keys_buffer.get_device_pointer();
    unsigned int* input_values  = m_values_buffer.get_device_pointer();
    unsigned int* output_keys   = m_temp_keys_buffer.get_device_pointer();
    unsigned int* output_values = m_temp_values_buffer.get_device_pointer();
    unsigned int* count_tables  = m_count_tables_buffer.get_device_pointer();

    // Perform radix sort in multiple passes
    static constexpr int NUM_PASSES = 32 / RADIX_SORT_RADIX_BITS;
    for (int pass = 0; pass < NUM_PASSES; pass++)
    {
        int bit_offset = pass * RADIX_SORT_RADIX_BITS;

        // Zero the count table before counting
        //
        // TODO memset with a kernel is faster?
        m_count_tables_buffer.memset_whole_buffer(0);

        // Count kernel: Count occurrences of each radix digit
        void* count_args[] = { &input_keys, &count_tables, &m_size, &bit_offset };
        m_count_kernel.launch_asynchronous(1024, 1, m_size, 1, count_args, m_stream);

        // Scan kernel: Compute exclusive prefix sum
        {
            std::vector<unsigned int> count_table_host = m_count_tables_buffer.download_data();

            std::cout << "Count table before: " << std::endl;
            for (int i = 0; i < count_table_host.size(); i++)
            {
                std::cout << count_table_host[i] << ", ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;

            // Replace the prefix scan here with just a block prefix scan because the count table is always small (256 elements) AND THIS IS A VERY VERY LONG
            // LINE BLAH LBAH TAKING MORE SPACE TO TRIGGER CLANGF FORAMT
            m_prefix_scan.upload_input_data(count_table_host);
            m_prefix_scan.scan();
        }

        m_count_tables_buffer.upload_data(m_prefix_scan.get_output_buffer().download_data());
        unsigned int* count_tables_prefix_scanned = m_count_tables_buffer.get_device_pointer();

        std::vector<unsigned int> after = m_count_tables_buffer.download_data();
        std::cout << "Count table after: " << std::endl;
        for (int i = 0; i < after.size(); i++)
        {
            std::cout << after[i] << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;

        // Reorder kernel: Scatter elements to sorted positions
        void* reorder_args[] = { &input_keys, &input_values, &output_keys, &output_values, &count_tables_prefix_scanned, &m_size, &bit_offset };
        m_reorder_kernel.launch_asynchronous(RADIX_SORT_THREADS_PER_BLOCK, 1, m_size, 1, reorder_args, m_stream);

        // Synchronize stream to ensure all kernels complete before next pass
        OROCHI_CHECK_ERROR(oroStreamSynchronize(m_stream));

        // Swap buffers for next pass
        if (pass < NUM_PASSES - 1)
        {
            // Swap the buffer objects, not just pointers
            std::swap(m_keys_buffer, m_temp_keys_buffer);
            std::swap(m_values_buffer, m_temp_values_buffer);

            // Update pointers for next iteration
            input_keys    = m_keys_buffer.get_device_pointer();
            input_values  = m_values_buffer.get_device_pointer();
            output_keys   = m_temp_keys_buffer.get_device_pointer();
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

OrochiBuffer<unsigned int>& RadixSort::get_sorted_keys_buffer()
{
    return m_keys_buffer;
}

OrochiBuffer<unsigned int>& RadixSort::get_sorted_values_buffer()
{
    return m_values_buffer;
}
