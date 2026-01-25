/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Renderer/Compute/ParallelPrefixScan.h"

#include <random>

ParallelPrefixScan::ParallelPrefixScan() : m_hiprt_ctx(nullptr), m_stream(nullptr), m_size_padded(0) {}

ParallelPrefixScan::ParallelPrefixScan(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
	: m_hiprt_ctx(hiprt_ctx), m_stream(stream), m_size_padded(0)
{
	initialize_kernels();
}

bool ParallelPrefixScan::is_setup()
{
	return m_hiprt_ctx != nullptr && m_stream != nullptr;
}

void ParallelPrefixScan::set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	m_hiprt_ctx = hiprt_ctx;
	m_stream = stream;

	initialize_kernels();
}

void ParallelPrefixScan::initialize_kernels()
{
	m_block_scan_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelPrefixScan/BlockScan.h");
	m_block_scan_kernel.set_kernel_function_name("ParallelPrefixScan_BlockScan");
	m_block_scan_kernel.compile(m_hiprt_ctx, {}, true, false);

	m_block_increment_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelPrefixScan/BlockIncrement.h");
	m_block_increment_kernel.set_kernel_function_name("ParallelPrefixScan_BlockIncrement");
	m_block_increment_kernel.compile(m_hiprt_ctx, {}, true, false);
}

void ParallelPrefixScan::upload_input_data(const std::vector<unsigned int>& data)
{
	if (data.size() > 0xFFFFFFFFull)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: input data size is too large (max supported size is 4,294,967,295 elements)");

		return;
	}

	// Padding the data size to be multiple of PARALLEL_PREFIX_SCAN_CHUNK_SIZE
	unsigned int padded_size = ((data.size() + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE) * PARALLEL_PREFIX_SCAN_CHUNK_SIZE;

	unsigned int divided_once_size = padded_size / PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
	unsigned int padded_divided_once_size = ((divided_once_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE) * PARALLEL_PREFIX_SCAN_CHUNK_SIZE;

	unsigned int divided_twice_size = padded_divided_once_size / PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
	unsigned int padded_divided_twice_size = ((divided_twice_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE) * PARALLEL_PREFIX_SCAN_CHUNK_SIZE;

	unsigned int divided_thrice_size = padded_divided_twice_size / PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
	unsigned int padded_divided_thrice_size = ((divided_thrice_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE) * PARALLEL_PREFIX_SCAN_CHUNK_SIZE;

	m_size_padded = padded_size;
	m_size_non_padded = data.size();

	m_input_buffer.resize(padded_size);
	m_input_buffer.upload_data(data);

	m_hierarchy_levels_used = 0;
	if (padded_size > PARALLEL_PREFIX_SCAN_CHUNK_SIZE)
	{
		m_level_0_block_sums.resize(std::max(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, padded_divided_once_size));
		m_scanned_level_0_blocks_sums.resize(std::max(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, padded_divided_once_size));

		m_hierarchy_levels_used++;
	}

	if (padded_size > PARALLEL_PREFIX_SCAN_CHUNK_SIZE * PARALLEL_PREFIX_SCAN_CHUNK_SIZE)
	{
		m_level_1_block_sums.resize(std::max(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, padded_divided_twice_size));
		m_scanned_level_1_blocks_sums.resize(std::max(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, padded_divided_twice_size));

		m_hierarchy_levels_used++;
	}

	if (padded_size > PARALLEL_PREFIX_SCAN_CHUNK_SIZE * PARALLEL_PREFIX_SCAN_CHUNK_SIZE * PARALLEL_PREFIX_SCAN_CHUNK_SIZE)
	{
		m_level_2_block_sums.resize(std::max(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, padded_divided_thrice_size));
		m_scanned_level_2_blocks_sums.resize(std::max(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, padded_divided_thrice_size));

		m_hierarchy_levels_used++;
	}

	m_output_buffer.resize(data.size());
}

void ParallelPrefixScan::scan()
{
	if (m_size_padded == 0 || !m_hiprt_ctx || !m_stream)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: scan() called but the class hasn't been setup properly or no data uploaded");

		return;
	}

	if (m_hierarchy_levels_used == 0)
	{
		// Simple case, input size fits in a single block scan

		unsigned int* input_data = m_input_buffer.get_device_pointer();
		unsigned int* output_data = m_output_buffer.get_device_pointer();
		unsigned int* zero_ptr = nullptr;

		void* scan_args[] = { &input_data, &output_data, &zero_ptr, &m_size_padded };
		m_block_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, (m_size_padded + 1) / 2, 1, scan_args, m_stream);
	}
	else
	{
		// More than 1 level

		/**
		 * Scanning the input in chunks
		 */
		{
			unsigned int* input_data = m_input_buffer.get_device_pointer();
			unsigned int* output_data = m_output_buffer.get_device_pointer();
			unsigned int* level_0_block_sums = m_level_0_block_sums.get_device_pointer();

			void* scan_args[] = { &input_data, &output_data, &level_0_block_sums, &m_size_padded };
			m_block_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, (m_size_padded + 1) / 2, 1, scan_args, m_stream);
		}

		/**
		 * Exclusive prefix-scanning the block sums of level 0
		 */
		 // We have scanned the input chunks, we need to scan the block sums now
		{
			unsigned int* level_0_block_sums = m_level_0_block_sums.get_device_pointer();
			unsigned int* scanned_level_0_block_sums = m_scanned_level_0_blocks_sums.get_device_pointer();
			unsigned int* level_1_block_sums = m_hierarchy_levels_used > 1 ? m_level_1_block_sums.get_device_pointer() : nullptr;
			unsigned int level_0_block_sums_size = m_level_0_block_sums.size();

			void* scan_block_sums_args[] = { &level_0_block_sums, &scanned_level_0_block_sums, &level_1_block_sums, &level_0_block_sums_size };
			m_block_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, (level_0_block_sums_size + 1) / 2, 1, scan_block_sums_args, m_stream);
		}

		if (m_hierarchy_levels_used > 1)
		{
			// We now need to scan the sums of level 1
			{
				unsigned int* level_1_block_sums = m_level_1_block_sums.get_device_pointer();
				unsigned int* scanned_level_1_block_sums = m_scanned_level_1_blocks_sums.get_device_pointer();
				unsigned int* level_2_block_sums = m_hierarchy_levels_used > 2 ? m_level_2_block_sums.get_device_pointer() : nullptr;
				unsigned int level_1_block_sums_size = m_level_1_block_sums.size();

				void* scan_block_sums_args[] = { &level_1_block_sums, &scanned_level_1_block_sums, &level_2_block_sums, &level_1_block_sums_size };
				m_block_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, (level_1_block_sums_size + 1) / 2, 1, scan_block_sums_args, m_stream);
			}

			if (m_hierarchy_levels_used > 2)
			{
				{
					// Scanning level_2_block_sums now

					unsigned int* level_2_block_sums = m_level_2_block_sums.get_device_pointer();
					unsigned int* scanned_level_2_block_sums = m_scanned_level_2_blocks_sums.get_device_pointer();
					unsigned int* level_3_block_sums = nullptr; // No support for more levels than that
					unsigned int level_2_block_sums_size = m_level_2_block_sums.size();

					void* scan_block_sums_args[] = { &level_2_block_sums, &scanned_level_2_block_sums, &level_3_block_sums, &level_2_block_sums_size };
					m_block_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, scan_block_sums_args, m_stream);
				}

				{
					// Adding the scanned level 2 blocks sums to the level 1 block sums

					unsigned int* scanned_level_1_block_sums = m_scanned_level_1_blocks_sums.get_device_pointer();
					unsigned int* scanned_level_2_block_sums = m_scanned_level_2_blocks_sums.get_device_pointer();
					unsigned int scanned_level_1_blocks_sums_size = m_scanned_level_1_blocks_sums.size();

					void* increment_args[] = { &scanned_level_1_block_sums, &scanned_level_2_block_sums, &scanned_level_1_blocks_sums_size };
					m_block_increment_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, scanned_level_1_blocks_sums_size, 1, increment_args, m_stream);
				}
			}

			{
				// Adding the scanned level 1 blocks sums to the level 0 block sums

				unsigned int* scanned_level_0_block_sums = m_scanned_level_0_blocks_sums.get_device_pointer();
				unsigned int* scanned_level_1_block_sums = m_scanned_level_1_blocks_sums.get_device_pointer();
				unsigned int scanned_level_0_blocks_sums_size = m_scanned_level_0_blocks_sums.size();

				void* increment_args[] = { &scanned_level_0_block_sums, &scanned_level_1_block_sums, &scanned_level_0_blocks_sums_size };
				m_block_increment_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, scanned_level_0_blocks_sums_size, 1, increment_args, m_stream);
			}
		}

		// We now have the scanned level 0 block sums and we have to add
		// them to 'output_data' which is the scanned (in chunks) input
		unsigned int* output_data = m_output_buffer.get_device_pointer();
		unsigned int* scanned_level_0_block_sums = m_scanned_level_0_blocks_sums.get_device_pointer();

		void* increment_args[] = { &output_data, &scanned_level_0_block_sums, &m_size_non_padded };
		m_block_increment_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, m_size_non_padded, 1, increment_args, m_stream);
	}
}

OrochiBuffer<unsigned int>& ParallelPrefixScan::get_output_buffer()
{
	return m_output_buffer;
}

void ParallelPrefixScan::unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelPrefixScan scanner(hiprt_ctx, stream);

	GPUKernel& m_block_scan_kernel = scanner.m_block_scan_kernel;

	// 100 Tests for the block scan
	std::mt19937 rng(42);
	for (int i = 0; i < 100; i++)
	{
		unsigned int input_size_original = std::max((unsigned int)PARALLEL_PREFIX_SCAN_CHUNK_SIZE, static_cast<unsigned int>(PARALLEL_PREFIX_SCAN_CHUNK_SIZE * (rng() % 100u)));

		unsigned int padded_size = ((input_size_original + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE) * PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
		unsigned int input_size = padded_size;

		std::vector<unsigned int> input(input_size);
		std::transform(input.begin(), input.end(), input.begin(), [](unsigned int) { return rand() % 100; });

		unsigned int running_sum = 0;
		std::vector<unsigned int> expected_output(input_size);
		std::vector<unsigned int> expected_block_sums(input_size / PARALLEL_PREFIX_SCAN_CHUNK_SIZE);
		for (size_t j = 0; j < input_size_original; j++)
		{
			if (j % PARALLEL_PREFIX_SCAN_CHUNK_SIZE == 0)
				running_sum = 0;
			else if (j % PARALLEL_PREFIX_SCAN_CHUNK_SIZE == PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1)
				expected_block_sums[j / PARALLEL_PREFIX_SCAN_CHUNK_SIZE] = running_sum + input[j];

			expected_output[j] = running_sum;
			running_sum += input[j];
		}

		OrochiBuffer<unsigned int> input_buffer(input);
		OrochiBuffer<unsigned int> output_buffer(input_size);
		OrochiBuffer<unsigned int> block_sums(input_size / PARALLEL_PREFIX_SCAN_CHUNK_SIZE);

		unsigned int* input_data = input_buffer.get_device_pointer();
		unsigned int* output_data = output_buffer.get_device_pointer();
		unsigned int* block_sums_data = block_sums.get_device_pointer();

		void* scan_args[] = { &input_data, &output_data, &block_sums_data, &input_size };
		m_block_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 2, 1, input_size / 2, 1, scan_args, stream);

		std::vector<unsigned int> output = output_buffer.download_data();
		std::vector<unsigned int> block_sums_output = block_sums.download_data();
		if (!std::equal(output.begin(), output.begin() + input_size_original, expected_output.begin()))
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan unit test failed for block scan test %d, size = %u", i, input_size_original);
			return;
		}

		if (!std::equal(block_sums_output.begin(), block_sums_output.end(), expected_block_sums.begin()))
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan unit test failed for block sums test %d, size = %u", i, input_size_original);
			return;
		}
	}

	// Full tests with random sizes
	oroEvent_t scan_start;
	oroEvent_t scan_end;

	OROCHI_CHECK_ERROR(oroEventCreate(&scan_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&scan_end));

	for (int i = 0; i < 100; i++)
	{
		rng.seed(i);

		unsigned int test_size = rng() % (65536 * 256 * 32) + 1;

		unsigned int running_sum = 0;
		std::vector<unsigned int> expected_output(test_size);
		std::vector<unsigned int> input(test_size);

		std::transform(input.begin(), input.end(), input.begin(), [&rng](unsigned int) { return rng() % 3; });

		auto start = std::chrono::high_resolution_clock::now();
		for (size_t j = 0; j < test_size; j++)
		{
			expected_output[j] = running_sum;
			running_sum += input[j];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms for " << test_size << " elements." << std::endl;

		scanner.upload_input_data(input);

		OROCHI_CHECK_ERROR(oroEventRecord(scan_start, stream));
		unsigned int repeats = 3;
		for (int i = 0; i < repeats; i++)
		{
			scanner.scan();
		}
		OROCHI_CHECK_ERROR(oroEventRecord(scan_end, stream));

		float elapsed_time_ms = 0.0f;
		OROCHI_CHECK_ERROR(oroEventSynchronize(scan_end));
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&elapsed_time_ms, scan_start, scan_end));

		// 21GItems to beat
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "\tParallelPrefixScan unit test %d: scanned %u elements in %.3f ms. %.3f GItems/s", i, test_size, elapsed_time_ms / repeats, (float)test_size / (elapsed_time_ms * 1e6f / repeats));

		std::vector<unsigned int> output = scanner.get_output_buffer().download_data();

#pragma omp parallel for
		for (long long int j = 0; j < test_size; j++)
		{
			if (output[j] != expected_output[j])
			{
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan unit test failed for test %d at index %lld (size=%u): got %u, expected %u", i, j, test_size, output[j], expected_output[j]);
			}
		}
	}

	OROCHI_CHECK_ERROR(oroEventDestroy(scan_start));
	OROCHI_CHECK_ERROR(oroEventDestroy(scan_end));
}
