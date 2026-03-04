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

RadixSort::RadixSort() : m_hiprt_ctx(nullptr), m_stream(nullptr), m_size(0) {}

RadixSort::RadixSort(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	set_context(hiprt_ctx, stream);
}

void RadixSort::set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	m_hiprt_ctx = hiprt_ctx;
	m_stream	= stream;

	m_prefix_scan.set_context(hiprt_ctx, stream);

	initialize_kernels();
}

void RadixSort::initialize_kernels()
{
	m_count_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Count.h");
	m_count_kernel.set_kernel_function_name("RadixSort_Count");
	m_count_kernel.compile(m_hiprt_ctx, {}, true, false);

	m_per_block_prefix_scan_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/PerBlockScan.h");
	m_per_block_prefix_scan_kernel.set_kernel_function_name("RadixSort_PerBlockScan");
	m_per_block_prefix_scan_kernel.compile(m_hiprt_ctx, {}, true, false);

	m_reorder_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/RadixSort/Reorder.h");
	m_reorder_kernel.set_kernel_function_name("RadixSort_Reorder");
	m_reorder_kernel.compile(m_hiprt_ctx, {}, true, false);
}

void RadixSort::upload_input_data(const std::vector<unsigned int>& keys, const std::vector<unsigned int>& values)
{
	if (keys.size() != values.size())
		return; // Error: sizes don't match
	else if (keys.size() == 0)
		return;

	m_size = keys.size();
	m_keys_buffer.resize(m_size);
	m_temp_keys_buffer.resize(m_size);
	m_values_buffer.resize(m_size);
	m_temp_values_buffer.resize(m_size);

	m_keys_buffer.upload_data(reinterpret_cast<const unsigned int*>(keys.data()));
	m_values_buffer.upload_data(reinterpret_cast<const unsigned int*>(values.data()));

	unsigned int per_block_count_table_size = (m_size + RADIX_SORT_INPUT_CHUNK_SIZE - 1) / RADIX_SORT_INPUT_CHUNK_SIZE * RADIX_SORT_RADIX_SIZE;
	if (m_per_block_count_tables_buffer.size() != per_block_count_table_size)
	{
		m_per_block_count_tables_buffer.resize(per_block_count_table_size);
		m_per_block_count_tables_scanned_buffer.resize(per_block_count_table_size);
	}

	unsigned int count_table_size = RADIX_SORT_RADIX_SIZE;
	if (m_global_count_tables_buffer.size() != count_table_size)
		m_global_count_tables_buffer.resize(count_table_size);
}

void RadixSort::sort()
{
	if (m_size == 0 || !m_hiprt_ctx || !m_stream)
		return;

	unsigned int* input_keys					 = m_keys_buffer.get_device_pointer();
	unsigned int* input_values					 = m_values_buffer.get_device_pointer();
	unsigned int* output_keys					 = m_temp_keys_buffer.get_device_pointer();
	unsigned int* output_values					 = m_temp_values_buffer.get_device_pointer();
	unsigned int* count_tables					 = m_global_count_tables_buffer.get_device_pointer();
	unsigned int* per_block_count_tables		 = m_per_block_count_tables_buffer.get_device_pointer();
	unsigned int* per_block_count_tables_scanned = m_per_block_count_tables_scanned_buffer.get_device_pointer();

	// Perform radix sort in multiple passes
	static constexpr int NUM_PASSES = 32 / RADIX_SORT_RADIX_BITS;
	for (int pass = 0; pass < NUM_PASSES; pass++)
	{
		int bit_offset = pass * RADIX_SORT_RADIX_BITS;

		// Zero the count table before counting
		//
		// TODO memset with a kernel is faster?
		m_global_count_tables_buffer.memset_whole_buffer(0);
		m_per_block_count_tables_buffer.memset_whole_buffer(0);
		m_per_block_count_tables_scanned_buffer.memset_whole_buffer(0);

		// Count kernel: Count occurrences of each radix digit
		void* count_args[] = { &input_keys, &count_tables, &per_block_count_tables, &m_size, &bit_offset };
		m_count_kernel.launch_asynchronous(RADIX_SORT_INPUT_CHUNK_SIZE, 1, m_size, 1, count_args, m_stream);

		// TODO this kernel is very un-optimal
		unsigned int num_blocks		= (m_size + RADIX_SORT_INPUT_CHUNK_SIZE - 1) / RADIX_SORT_INPUT_CHUNK_SIZE;
		void* per_block_scan_args[] = { &per_block_count_tables, &per_block_count_tables_scanned, &num_blocks };
		m_per_block_prefix_scan_kernel.launch_asynchronous(RADIX_SORT_RADIX_SIZE, 1, RADIX_SORT_RADIX_SIZE, 1, per_block_scan_args, m_stream);

		// Scan kernel: Compute exclusive prefix sum
		{
			// DEBUG VERIFICATION BLOCK
			{
				std::vector<unsigned int> count_table_host					 = m_global_count_tables_buffer.download_data();
				std::vector<unsigned int> per_block_count_table_host		 = m_per_block_count_tables_buffer.download_data();
				std::vector<unsigned int> per_block_count_table_scanned_host = m_per_block_count_tables_scanned_buffer.download_data();
				std::vector<unsigned int> input_keys_host					 = m_keys_buffer.download_data();
				std::vector<unsigned int> expected_count_table(RADIX_SORT_RADIX_SIZE, 0);

				for (int i = 0; i < m_size; i++)
				{
					unsigned int key   = input_keys_host.at(i);
					unsigned int digit = (key >> bit_offset) & (RADIX_SORT_RADIX_SIZE - 1);

					expected_count_table[digit]++;
				}

				for (int i = 0; i < RADIX_SORT_RADIX_SIZE; i++)
				{
					if (count_table_host.at(i) != expected_count_table.at(i))
					{
						std::cout << "Count table mismatch at index " << i << ": got " << count_table_host[i] << ", expected " << expected_count_table[i]
								  << std::endl;

						break;
					}
				}

				unsigned int num_blocks = (input_keys_host.size() + RADIX_SORT_INPUT_CHUNK_SIZE - 1) / RADIX_SORT_INPUT_CHUNK_SIZE;
				std::vector<unsigned int> per_block_expected_count_table(num_blocks * RADIX_SORT_RADIX_SIZE, 0);
				for (int block = 0; block < num_blocks; block++)
				{
					for (int i = 0; i < RADIX_SORT_INPUT_CHUNK_SIZE; i++)
					{
						unsigned int index = block * RADIX_SORT_INPUT_CHUNK_SIZE + i;
						if (index >= m_size)
							break;

						unsigned int key   = input_keys_host[index];
						unsigned int digit = (key >> bit_offset) & RADIX_SORT_RADIX_MASK;

						// Writing in column major order so that the prefix scan can be done easily
						per_block_expected_count_table.at(block * RADIX_SORT_RADIX_SIZE + digit)++;
					}
				}

				std::vector<unsigned int> per_block_expected_count_table_prefix_summed(num_blocks * RADIX_SORT_RADIX_SIZE, 0);
				for (int digit = 0; digit < RADIX_SORT_RADIX_SIZE; digit++)
				{
					unsigned int running_sum = 0;

					for (int block = 0; block < num_blocks; block++)
					{
						unsigned int index = block * RADIX_SORT_RADIX_SIZE + digit;
						unsigned int temp  = per_block_expected_count_table.at(index);

						per_block_expected_count_table_prefix_summed.at(index) = running_sum;
						running_sum += temp;
					}

					running_sum = 0;
				}

				for (int i = 0; i < per_block_count_table_scanned_host.size(); i++)
				{
					if (per_block_count_table_scanned_host.at(i) != per_block_expected_count_table_prefix_summed.at(i))
					{
						std::cout << "Per-block count table mismatch at index " << i << ": got " << per_block_count_table_scanned_host.at(i) << ", expected "
								  << per_block_expected_count_table_prefix_summed.at(i) << std::endl;

						break;
					}
				}
			}

			// Replace the prefix scan here with just a block prefix scan because the count table is always small (256 elements)
			std::vector<unsigned int> count_table_host = m_global_count_tables_buffer.download_data();

			// TODO Do not upload data but just pass the pointer directly to the prefix scan
			m_prefix_scan.upload_input_data(count_table_host);
			m_prefix_scan.scan();
		}

		// TODO Avoid this download + upload by directly using the output buffer of the prefix scan
		m_global_count_tables_buffer.upload_data(m_prefix_scan.get_output_buffer().download_data());
		unsigned int* global_count_table_prefix_scanned	   = m_global_count_tables_buffer.get_device_pointer();
		unsigned int* per_block_count_table_prefix_scanned = m_per_block_count_tables_scanned_buffer.get_device_pointer();

		// Reorder kernel: Scatter elements to sorted positions
		void* reorder_args[] = {
			&input_keys, &input_values, &output_keys, &output_values, &global_count_table_prefix_scanned, &per_block_count_table_prefix_scanned,
			&m_size,	 &bit_offset
		};
		m_reorder_kernel.launch_asynchronous(RADIX_SORT_INPUT_CHUNK_SIZE, 1, m_size, 1, reorder_args, m_stream);

		// Synchronize stream to ensure all kernels complete before next pass
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_stream));

		std::vector<unsigned int> input_keys_host			= m_keys_buffer.download_data();
		std::vector<unsigned int> input_keys_host_reordered = m_temp_keys_buffer.download_data();

		// Swap buffers for next pass
		if (pass < NUM_PASSES - 1)
		{
			// Swap the buffer objects, not just pointers
			std::swap(m_keys_buffer, m_temp_keys_buffer);
			std::swap(m_values_buffer, m_temp_values_buffer);

			// Update pointers for next iteration
			input_keys	  = m_keys_buffer.get_device_pointer();
			input_values  = m_values_buffer.get_device_pointer();
			output_keys	  = m_temp_keys_buffer.get_device_pointer();
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

void RadixSort::unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	RadixSort sorter(hiprt_ctx, stream);

	std::mt19937 rng(42);
	// Tests with random sizes

	oroEvent_t scan_start;
	oroEvent_t scan_end;

	OROCHI_CHECK_ERROR(oroEventCreate(&scan_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&scan_end));

	for (int i = 0; i < 100; i++)
	{
		rng.seed(i);

		unsigned int test_size = rng() % 65536;

		std::vector<unsigned int> input_keys(test_size);
		std::vector<unsigned int> input_values(test_size);

		std::iota(input_values.begin(), input_values.end(), 0);
		std::transform(input_keys.begin(), input_keys.end(), input_keys.begin(), [&rng](unsigned int) { return rng() % 10; });

		std::vector<std::pair<unsigned int, unsigned int>> key_value_pairs(test_size);

		for (size_t j = 0; j < test_size; j++)
		{
			key_value_pairs[j] = { input_keys[j], input_values[j] };
		}

		auto start = std::chrono::high_resolution_clock::now();
		std::stable_sort(key_value_pairs.begin(), key_value_pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
		auto stop = std::chrono::high_resolution_clock::now();

		std::cout << "CPU sort time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms for " << test_size << " elements."
				  << std::endl;

		sorter.upload_input_data(input_keys, input_values);

		OROCHI_CHECK_ERROR(oroEventRecord(scan_start, stream));
		unsigned int repeats = 1;
		for (int j = 0; j < repeats; j++)
		{
			sorter.sort();
		}
		OROCHI_CHECK_ERROR(oroEventRecord(scan_end, stream));

		float elapsed_time_ms = 0.0f;
		OROCHI_CHECK_ERROR(oroEventSynchronize(scan_end));
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&elapsed_time_ms, scan_start, scan_end));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "\tRadixSort unit test %d: sorted %u elements in %.3f ms. %.3fGItems/s", i, test_size,
								elapsed_time_ms / repeats, test_size / (elapsed_time_ms * 1e6f / repeats));

		std::vector<unsigned int> sorted_keys	= sorter.get_sorted_keys_buffer().download_data();
		std::vector<unsigned int> sorted_values = sorter.get_sorted_values_buffer().download_data();

		for (size_t j = 0; j < test_size; j++)
		{
			if (sorted_keys[j] != key_value_pairs[j].first || sorted_values[j] != key_value_pairs[j].second)
			{
				std::cout << "Mismatch at index " << j << ": got (" << sorted_keys[j] << ", " << sorted_values[j] << "), expected (" << key_value_pairs[j].first
						  << ", " << key_value_pairs[j].second << ")" << std::endl;

				break;
			}
		}
	}
}
