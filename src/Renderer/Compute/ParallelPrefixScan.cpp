/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Renderer/Compute/ParallelPrefixScan.h"

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

void ParallelPrefixScan::upload_data(const std::vector<unsigned int>& data)
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

#define DEBUG 1

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
		/*if (m_hierarchy_levels_used == 1)
		{*/
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

			if (DEBUG)
			{
				std::vector<unsigned int> debug_input = m_input_buffer.download_data();
				std::vector<unsigned int> debug_output = m_output_buffer.download_data();
				std::vector<unsigned int> level_0_block_sums_host = m_level_0_block_sums.download_data();
				std::vector<unsigned int> expected_level_0_block_sums(level_0_block_sums_host.size());
				std::vector<unsigned int> expected_output(debug_output.size());

				for (size_t i = 0; i < m_input_buffer.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
				{
					for (int j = 0; j < PARALLEL_PREFIX_SCAN_CHUNK_SIZE; j++)
						expected_level_0_block_sums.at(i) += debug_input[i * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + j];
				}

				unsigned int running_sum = 0;
				for (size_t i = 0; i < debug_output.size(); i++)
				{
					if (i % PARALLEL_PREFIX_SCAN_CHUNK_SIZE == 0)
						running_sum = 0;
					expected_output.at(i) = running_sum;
					running_sum += debug_input.at(i);
				}

				for (size_t i = 0; i < m_size_padded / 256; i++)
				{
					if (level_0_block_sums_host.at(i) != expected_level_0_block_sums.at(i))
					{
						g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 0 block sums mismatch at index %zu: got %u, expected %u", i, level_0_block_sums_host.at(i), expected_level_0_block_sums.at(i));
						return;
					}
				}

				for (size_t i = 0; i < debug_output.size(); i++)
				{
					if (debug_output.at(i) != expected_output.at(i))
					{
						g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Output chunk scan mismatch at index %zu: got %u, expected %u", i, debug_output.at(i), expected_output.at(i));
						return;
					}
				}
			}
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

			if (DEBUG)
			{
				if (level_1_block_sums != nullptr)
				{
					std::vector<unsigned int> input_host = m_level_0_block_sums.download_data();
					std::vector<unsigned int> output_host = m_scanned_level_0_blocks_sums.download_data();
					std::vector<unsigned int> m_level_1_block_sums_host = m_level_1_block_sums.download_data();
					std::vector<unsigned int> expected_level_1_block_sums(m_level_1_block_sums.size());
					std::vector<unsigned int> expected_output_host(output_host.size());

					for (size_t i = 0; i < m_level_0_block_sums.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
					{
						for (int j = 0; j < PARALLEL_PREFIX_SCAN_CHUNK_SIZE; j++)
							expected_level_1_block_sums.at(i) += input_host[i * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + j];
					}

					unsigned int running_sum = 0;
					for (size_t i = 0; i < output_host.size(); i++)
					{
						if (i % PARALLEL_PREFIX_SCAN_CHUNK_SIZE == 0)
							running_sum = 0;

						expected_output_host.at(i) = running_sum;
						running_sum += input_host.at(i);
					}

					for (size_t i = 0; i < m_level_0_block_sums.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
					{
						if (m_level_1_block_sums_host.at(i) != expected_level_1_block_sums.at(i))
						{
							g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 1 block sums mismatch at index %zu: got %u, expected %u", i, m_level_1_block_sums_host.at(i), expected_level_1_block_sums.at(i));
							return;
						}
					}

					for (size_t i = 0; i < output_host.size(); i++)
					{
						if (output_host.at(i) != expected_output_host.at(i))
						{
							g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 0 block sums scan mismatch at index %zu: got %u, expected %u", i, output_host.at(i), expected_output_host.at(i));
							return;
						}
					}
				}
			}
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

				if (DEBUG)
				{
					if (m_hierarchy_levels_used > 2)
					{
						std::vector<unsigned int> debug_level_1_block_sums = m_level_1_block_sums.download_data();
						std::vector<unsigned int> expected_level_2_block_sums(m_level_2_block_sums.size());
						for (size_t i = 0; i < m_level_1_block_sums.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
						{
							for (int j = 0; j < PARALLEL_PREFIX_SCAN_CHUNK_SIZE; j++)
								expected_level_2_block_sums.at(i) += debug_level_1_block_sums[i * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + j];
						}

						std::vector<unsigned int> level_2_block_sums_downloaded = m_level_2_block_sums.download_data();
						for (size_t i = 0; i < debug_level_1_block_sums.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
						{
							if (level_2_block_sums_downloaded.at(i) != expected_level_2_block_sums.at(i))
							{
								g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 1 block sums scan mismatch at index %zu: got %u, expected %u", i, debug_level_1_block_sums.at(i), expected_level_2_block_sums.at(i));

								return;
							}
						}
					}
				}
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

					if (DEBUG)
					{
						std::vector<unsigned int> input_host = m_level_2_block_sums.download_data();
						std::vector<unsigned int> expected_output_host(input_host.size());
						unsigned int running_sum = 0;
						for (size_t i = 0; i < m_level_1_block_sums.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
						{
							expected_output_host.at(i) = running_sum;
							running_sum += input_host.at(i);
						}

						std::vector<unsigned int> output_host = m_scanned_level_2_blocks_sums.download_data();
						for (size_t i = 0; i < m_level_1_block_sums.size() / PARALLEL_PREFIX_SCAN_CHUNK_SIZE; i++)
						{
							if (output_host.at(i) != expected_output_host.at(i))
							{
								g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 2 block sums scan mismatch at index %zu: got %u, expected %u", i, output_host.at(i), expected_output_host.at(i));
								return;
							}
						}
					}
				}

				{
					// Adding the scanned level 2 blocks sums to the level 1 block sums

					unsigned int* scanned_level_1_block_sums = m_scanned_level_1_blocks_sums.get_device_pointer();
					unsigned int* scanned_level_2_block_sums = m_scanned_level_2_blocks_sums.get_device_pointer();
					unsigned int scanned_level_1_blocks_sums_size = m_scanned_level_1_blocks_sums.size();

					void* increment_args[] = { &scanned_level_1_block_sums, &scanned_level_2_block_sums, &scanned_level_1_blocks_sums_size };
					std::vector<unsigned int> input_host;
					if (DEBUG)
						input_host = m_scanned_level_1_blocks_sums.download_data();
					m_block_increment_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, scanned_level_1_blocks_sums_size, 1, increment_args, m_stream);

					if (DEBUG)
					{
						std::vector<unsigned int> increments_host = m_scanned_level_2_blocks_sums.download_data();
						std::vector<unsigned int> expected_output_host(input_host.size());
						for (size_t i = 0; i < input_host.size(); i++)
						{
							unsigned int increment = increments_host[i / PARALLEL_PREFIX_SCAN_CHUNK_SIZE];
							expected_output_host.at(i) = input_host.at(i) + increment;
						}

						std::vector<unsigned int> output_host = m_scanned_level_1_blocks_sums.download_data();

						for (size_t i = 0; i < input_host.size(); i++)
						{
							if (output_host.at(i) != expected_output_host.at(i))
							{
								g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 1 block sums increment mismatch at index %zu: got %u, expected %u", i, output_host.at(i), expected_output_host.at(i));
								return;
							}
						}
					}
				}
			}

			{
				// Adding the scanned level 1 blocks sums to the level 0 block sums

				unsigned int* scanned_level_0_block_sums = m_scanned_level_0_blocks_sums.get_device_pointer();
				unsigned int* scanned_level_1_block_sums = m_scanned_level_1_blocks_sums.get_device_pointer();
				unsigned int scanned_level_0_blocks_sums_size = m_scanned_level_0_blocks_sums.size();

				void* increment_args[] = { &scanned_level_0_block_sums, &scanned_level_1_block_sums, &scanned_level_0_blocks_sums_size };
				std::vector<unsigned int> input_host;
				if (DEBUG)
					input_host = m_scanned_level_0_blocks_sums.download_data();
				m_block_increment_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, scanned_level_0_blocks_sums_size, 1, increment_args, m_stream);

				if (DEBUG)
				{
					std::vector<unsigned int> increments_host = m_scanned_level_1_blocks_sums.download_data();
					std::vector<unsigned int> expected_output_host(scanned_level_0_blocks_sums_size);
					std::vector<unsigned int> output_host = m_scanned_level_0_blocks_sums.download_data();

					for (size_t i = 0; i < scanned_level_0_blocks_sums_size; i++)
					{
						unsigned int increment = increments_host[i / PARALLEL_PREFIX_SCAN_CHUNK_SIZE];
						expected_output_host.at(i) = input_host.at(i) + increment;
					}

					for (size_t i = 0; i < scanned_level_0_blocks_sums_size; i++)
					{
						if (output_host.at(i) != expected_output_host.at(i))
						{
							g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Level 1 block sums increment mismatch at index %zu: got %u, expected %u", i, output_host.at(i), expected_output_host.at(i));
							return;
						}
					}
				}
			}
		}

		// We now have the scanned level 0 block sums and we have to add
		// them to 'output_data' which is the scanned (in chunks) input
		unsigned int* output_data = m_output_buffer.get_device_pointer();
		unsigned int* scanned_level_0_block_sums = m_scanned_level_0_blocks_sums.get_device_pointer();

		void* increment_args[] = { &output_data, &scanned_level_0_block_sums, &m_size_non_padded };
		std::vector<unsigned int> debug_output_before_increment;
		if (DEBUG)
			debug_output_before_increment = m_output_buffer.download_data();
		m_block_increment_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, m_size_non_padded, 1, increment_args, m_stream);

		if (DEBUG)
		{
			std::vector<unsigned int> increments = m_scanned_level_0_blocks_sums.download_data();
			std::vector<unsigned int> expected_output_host(m_output_buffer.size());
			for (size_t i = 0; i < m_size_non_padded; i++)
			{
				unsigned int increment = increments[i / PARALLEL_PREFIX_SCAN_CHUNK_SIZE];
				expected_output_host.at(i) = debug_output_before_increment.at(i) + increment;
			}

			std::vector<unsigned int> output_host = m_output_buffer.download_data();
			for (size_t i = 0; i < m_size_non_padded; i++)
			{
				if (output_host.at(i) != expected_output_host.at(i))
				{
					g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Final output mismatch at index %zu: got %u, expected %u", i, output_host.at(i), expected_output_host.at(i));
					return;
				}
			}
		}

		if (DEBUG)
		{
			std::vector<unsigned int> input_host = m_input_buffer.download_data();
			std::vector<unsigned int> expected_output_host(input_host.size());
			unsigned int running_sum = 0;
			for (size_t i = 0; i < m_size_non_padded; i++)
			{
				expected_output_host.at(i) = running_sum;
				running_sum += input_host.at(i);
			}

			std::vector<unsigned int> output_host = m_output_buffer.download_data();
			for (size_t i = 0; i < m_size_non_padded; i++)
			{
				if (output_host.at(i) != expected_output_host.at(i))
				{
					g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan: Final output mismatch at index %zu: got %u, expected %u", i, output_host.at(i), expected_output_host.at(i));
					return;
				}
			}
		}
	}
}

OrochiBuffer<unsigned int>& ParallelPrefixScan::get_output_buffer()
{
	return m_output_buffer;
}

#include <random>

void ParallelPrefixScan::unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelPrefixScan scanner(hiprt_ctx, stream);

	GPUKernel& m_block_scan_kernel = scanner.m_block_scan_kernel;

	// 100 Tests for the block scan
	std::mt19937 rng(42);
	for (int i = 0; i < 100; i++)
	{
		unsigned int input_size_original = std::max((unsigned int)PARALLEL_PREFIX_SCAN_CHUNK_SIZE, (unsigned int)PARALLEL_PREFIX_SCAN_CHUNK_SIZE * (rng() % 1000u));

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
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan unit test failed for block scan test %d", i);
			return;
		}

		if (!std::equal(block_sums_output.begin(), block_sums_output.end(), expected_block_sums.begin()))
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan unit test failed for block sums test %d", i);
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
		rng.seed(42);

		unsigned int test_size = rng() % (65536 * 256 * 32) + 1;

		std::vector<unsigned int> input(test_size);
		std::transform(input.begin(), input.end(), input.begin(), [&rng](unsigned int) { return rng() % 3; });

		unsigned int running_sum = 0;
		std::vector<unsigned int> expected_output(test_size);
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t j = 0; j < test_size; j++)
		{
			expected_output[j] = running_sum;
			running_sum += input[j];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms for " << test_size << " elements." << std::endl;

		scanner.upload_data(input);

		OROCHI_CHECK_ERROR(oroEventRecord(scan_start, stream));
		scanner.scan();
		OROCHI_CHECK_ERROR(oroEventRecord(scan_end, stream));

		float elapsed_time_ms = 0.0f;
		OROCHI_CHECK_ERROR(oroEventSynchronize(scan_end));
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&elapsed_time_ms, scan_start, scan_end));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "\tParallelPrefixScan unit test %d: scanned %u elements in %.3f ms", i, test_size, elapsed_time_ms);

		std::vector<unsigned int> output = scanner.get_output_buffer().download_data();

#pragma omp parallel for
		for (long long int j = 0; j < test_size; j++)
		{
			if (output[j] != expected_output[j])
			{
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScan unit test failed for test %d at index %d (size=%zu): got %u, expected %u", i, j, test_size, output[j], expected_output[j]);
				break;
			}
		}
	}

	OROCHI_CHECK_ERROR(oroEventDestroy(scan_start));
	OROCHI_CHECK_ERROR(oroEventDestroy(scan_end));
}
