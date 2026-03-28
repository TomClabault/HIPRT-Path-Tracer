/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"

#include <random>

ParallelPrefixScanDecoupledLookback::ParallelPrefixScanDecoupledLookback() : m_hiprt_ctx(nullptr), m_stream(nullptr), m_size(0) {}

ParallelPrefixScanDecoupledLookback::ParallelPrefixScanDecoupledLookback(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
	: m_hiprt_ctx(hiprt_ctx), m_stream(stream), m_size(0)
{
	initialize_kernels();
}

bool ParallelPrefixScanDecoupledLookback::is_setup()
{
	return m_hiprt_ctx != nullptr && m_stream != nullptr;
}

void ParallelPrefixScanDecoupledLookback::set_context(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	m_hiprt_ctx = hiprt_ctx;
	m_stream	= stream;

	initialize_kernels();
}

void ParallelPrefixScanDecoupledLookback::initialize_kernels()
{
	m_scan_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelPrefixScanDecoupledLookback/Scan.h");
	m_scan_kernel.set_kernel_function_name("ParallelPrefixScanDecoupledLookback_Scan");
	m_scan_kernel.compile(m_hiprt_ctx, {}, true, false);
	m_scan_kernel.set_measure_execution_time(false);

	m_block_descriptor_init_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelPrefixScanDecoupledLookback/BlockDescriptorInit.h");
	m_block_descriptor_init_kernel.set_kernel_function_name("ParallelPrefixScanDecoupledLookback_BlockDescriptorInit");
	m_block_descriptor_init_kernel.compile(m_hiprt_ctx, {}, true, false);
	m_block_descriptor_init_kernel.set_measure_execution_time(false);
}

void ParallelPrefixScanDecoupledLookback::resize(unsigned int element_count)
{
	m_last_resize_element_count = element_count;

	m_size = element_count;

	m_input_buffer.resize(m_size);
	m_output_buffer.resize(m_size);

	m_input_data_pointer  = m_input_buffer.get_device_pointer();
	m_output_data_pointer = m_output_buffer.get_device_pointer();

	m_global_block_index_counter_buffer.resize(1);
	m_block_descriptors_buffer.resize((m_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE);
}

void ParallelPrefixScanDecoupledLookback::upload_input_data(const std::vector<unsigned int>& data)
{
	m_size = data.size();

	resize(m_size);

	m_input_buffer.upload_data(data);

	m_input_data_pointer  = m_input_buffer.get_device_pointer();
	m_output_data_pointer = m_output_buffer.get_device_pointer();
}

void ParallelPrefixScanDecoupledLookback::set_data_pointers(unsigned int* input_buffer_pointer, unsigned int element_count)
{
	set_data_pointers(input_buffer_pointer, nullptr, element_count);
}

void ParallelPrefixScanDecoupledLookback::set_data_pointers(unsigned int* input_buffer_pointer, unsigned int* output_buffer_pointer, unsigned int element_count)
{
	if (m_last_resize_element_count != element_count)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
								"ParallelPrefixScanDecoupledLookback::set_data_pointers() called with an element_count (%u) that is different from the last "
								"one used in resize() (%u). This is "
								"invalid usage and will lead to undefined behavior.",
								element_count, m_last_resize_element_count);

		Debug::debugbreak();

		return;
	}

	m_size = element_count;

	m_input_data_pointer = input_buffer_pointer;
	if (output_buffer_pointer)
		m_output_data_pointer = output_buffer_pointer;
	else
		m_output_data_pointer = m_output_buffer.get_device_pointer();
}

void ParallelPrefixScanDecoupledLookback::scan(bool auto_stream_synchronize)
{
	if (m_size == 0 || !m_hiprt_ctx || !m_stream)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
								"ParallelPrefixScanDecoupledLookback: scan() called but the class hasn't been setup properly or no data uploaded");

		return;
	}

	ParallelPrefixScanDecoupledLookbackBlockDescriptor* block_descriptors_buffer_pointer = m_block_descriptors_buffer.get_device_pointer();

	unsigned int block_descriptor_count				 = m_block_descriptors_buffer.size();
	unsigned int* global_block_index_counter_pointer = m_global_block_index_counter_buffer.get_device_pointer();
	void* block_descriptor_init_args[]				 = { &block_descriptors_buffer_pointer, &block_descriptor_count, &global_block_index_counter_pointer };
	m_block_descriptor_init_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, block_descriptor_count, 1, block_descriptor_init_args, m_stream);

	unsigned int* input_buffer_pointer	= m_input_data_pointer;
	unsigned int* output_buffer_pointer = m_output_data_pointer;
	void* block_scan_args[] = { &input_buffer_pointer, &output_buffer_pointer, &block_descriptors_buffer_pointer, &global_block_index_counter_pointer,
								&m_size };
	m_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, m_size, 1, block_scan_args, m_stream);

	if (auto_stream_synchronize)
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_stream));
}

float ParallelPrefixScanDecoupledLookback::get_last_execution_time()
{
	m_block_descriptor_init_kernel.compute_execution_time();
	m_scan_kernel.compute_execution_time();

	return m_block_descriptor_init_kernel.get_last_execution_time() + m_scan_kernel.get_last_execution_time();
}

OrochiBuffer<unsigned int>& ParallelPrefixScanDecoupledLookback::get_output_buffer()
{
	return m_output_buffer;
}

void ParallelPrefixScanDecoupledLookback::unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelPrefixScanDecoupledLookback scanner(hiprt_ctx, stream);

	std::mt19937 rng(42);

	// Full tests with random sizes
	oroEvent_t scan_start;
	oroEvent_t scan_end;

	OROCHI_CHECK_ERROR(oroEventCreate(&scan_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&scan_end));

	for (int i = 0; i < 10; i++)
	{
		rng.seed(i);

		unsigned int test_size = rng() % 500000000;

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
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms for " << test_size << " elements."
				  << std::endl;

		scanner.upload_input_data(input);

		OROCHI_CHECK_ERROR(oroEventRecord(scan_start, stream));
		unsigned int repeats = 5;
		for (int j = 0; j < repeats; j++)
		{
			scanner.scan();
		}
		OROCHI_CHECK_ERROR(oroEventRecord(scan_end, stream));

		float elapsed_time_ms = 0.0f;
		OROCHI_CHECK_ERROR(oroEventSynchronize(scan_end));
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&elapsed_time_ms, scan_start, scan_end));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO,
								"\tParallelPrefixScanDecoupledLookback unit test %d: scanned %u elements in %.3f ms. %.3fGItems/s", i, test_size,
								elapsed_time_ms / repeats, test_size / (elapsed_time_ms * 1e6f / repeats));

		std::vector<unsigned int> output = scanner.get_output_buffer().download_data();

		for (long long int j = 0; j < test_size; j++)
		{
			if (output[j] != expected_output[j])
			{
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
										"ParallelPrefixScanDecoupledLookback unit test failed for test %d at index %lld (size=%u): got %u, expected %u", i, j,
										test_size, output[j], expected_output[j]);

				Debug::debugbreak();

				return;
			}
		}
	}

	OROCHI_CHECK_ERROR(oroEventDestroy(scan_start));
	OROCHI_CHECK_ERROR(oroEventDestroy(scan_end));
}
