/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/ParallelReductionCommon.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"
#include "Renderer/Compute/ParallelSegmentedReduction.h"

#include <bitset>
#include <functional>
#include <random>

template <typename T>
ParallelSegmentedReduction<T>::ParallelSegmentedReduction() : m_hiprt_ctx(nullptr), m_stream(nullptr), m_size(0)
{
}

template <typename T>
ParallelSegmentedReduction<T>::ParallelSegmentedReduction(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream) : m_size(0)
{
	init(hiprt_ctx, stream);
}

template <typename T>
void ParallelSegmentedReduction<T>::init(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	m_hiprt_ctx = hiprt_ctx;
	m_stream	= stream;

	m_segment_ids_prefix_scan.init(hiprt_ctx, stream);

	initialize_kernels();
}

template <typename T>
void ParallelSegmentedReduction<T>::initialize_kernels()
{
	m_scan_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelSegmentedReduction/Scan.h");
	m_scan_kernel.set_kernel_function_name("ParallelSegmentedReduction_Reduce");
	m_scan_kernel.get_kernel_options().set_string_macro_value("DataType", get_data_type_as_string());
	m_scan_kernel.set_measure_execution_time(false);

	set_input_id_transform(std::make_unique<ComputeInputIDTransformIdentity>());
	set_data_transform(std::make_unique<ComputeDataTransformIdentity>());

	class ComputeDataTransformSegmentIdsPrefixScan : public ComputeDataTransform
	{
	public:
		virtual std::string emit_input_transform() const override
		{
			return "return ((value >> (global_element_id & 31)) & 1) && global_element_id != 0;";
		}

		virtual std::string emit_output_transform() const override
		{
			return "return value;";
		}
	};

	m_segment_ids_prefix_scan.set_input_id_transform(std::make_unique<ComputeInputIDTransformUnpackUintBits>());
	m_segment_ids_prefix_scan.set_data_transform(std::make_unique<ComputeDataTransformSegmentIdsPrefixScan>());
	m_segment_ids_prefix_scan.set_exclusive_scan(false);
}

template <typename T>
void ParallelSegmentedReduction<T>::set_input_id_transform(std::unique_ptr<ComputeInputIDTransform> transform)
{
	m_scan_kernel.get_kernel_options().set_string_macro_value(ComputeInputIDTransform::INPUT_ID_TRANSFORM_STRING_STUB, transform->emit_input_id_transform());
}

template <typename T>
void ParallelSegmentedReduction<T>::set_data_transform(std::unique_ptr<ComputeDataTransform> transform)
{
	m_scan_kernel.get_kernel_options().set_string_macro_value(ComputeDataTransform::INPUT_TRANSFORM_STRING_STUB, transform->emit_input_transform());
	m_scan_kernel.get_kernel_options().set_string_macro_value(ComputeDataTransform::OUTPUT_TRANSFORM_STRING_STUB, transform->emit_output_transform());
}

template <typename T>
void ParallelSegmentedReduction<T>::compile()
{
	m_scan_kernel.compile(m_hiprt_ctx, {}, true, false);

	m_segment_ids_prefix_scan.compile();
}

template <typename T>
void ParallelSegmentedReduction<T>::resize(unsigned int element_count)
{
	if (m_last_resize_element_count == element_count)
		// Nothing to resize
		return;

	m_last_resize_element_count = element_count;

	m_size = element_count;

	m_input_buffer.resize(m_size);
	m_flags_buffer.resize((m_size + 31) / 32); // We need one flag bit per element, so we need ceil(size / 32) unsigned ints to store them
	m_output_buffer.resize(m_size);
	m_output_buffer.memset_whole_buffer(0);

	m_segment_ids_prefix_scan.resize(element_count);
}

template <typename T>
void ParallelSegmentedReduction<T>::free()
{
	m_input_buffer.free_no_error();
	m_flags_buffer.free_no_error();
	m_output_buffer.free_no_error();

	m_input_data_pointer = nullptr;
	m_flags_data_pointer = nullptr;

	m_size						= 0;
	m_last_resize_element_count = 0;
}

template <typename T>
void ParallelSegmentedReduction<T>::upload_input_data(const std::vector<T>& data, const std::vector<unsigned int>& flags)
{
	m_size = data.size();

	resize(data.size());

	m_input_buffer.upload_data(data);
	m_flags_buffer.upload_data(flags);

	m_input_data_pointer = m_input_buffer.get_device_pointer();
	m_flags_data_pointer = m_flags_buffer.get_device_pointer();

	m_segment_ids_prefix_scan.set_data_pointers(m_flags_data_pointer, data.size());
}

template <typename T>
void ParallelSegmentedReduction<T>::set_data_pointers(T* device_data_pointer, unsigned int element_count)
{
	set_data_pointers(device_data_pointer, nullptr, element_count);
}

template <typename T>
void ParallelSegmentedReduction<T>::set_data_pointers(T* device_data_pointer, unsigned int* device_flags_pointer, unsigned int element_count)
{
	if (m_last_resize_element_count < element_count)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
								"ParallelSegmentedReduction::set_data_pointers() called with an element_count (%u) that is different from the last one used "
								"in resize() (%u). This is invalid usage.",
								element_count, m_last_resize_element_count);

		Debug::debugbreak();

		return;
	}

	m_size = element_count;

	m_input_data_pointer = device_data_pointer;
	if (device_flags_pointer)
		m_flags_data_pointer = device_flags_pointer;
	m_segment_ids_prefix_scan.set_data_pointers(device_flags_pointer, element_count);
}

template <typename T>
unsigned int ParallelSegmentedReduction<T>::get_evenly_spaced_segment_size()
{
	return m_evenly_spaced_segment_size;
}

template <typename T>
void ParallelSegmentedReduction<T>::set_evenly_spaced_segment_size(unsigned int segment_size)
{
	m_evenly_spaced_segment_size = segment_size;
}

template <typename T>
void ParallelSegmentedReduction<T>::reduce(bool auto_stream_synchronize)
{
	if (!m_hiprt_ctx || !m_stream)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
								"ParallelPrefixScanDecoupledLookback::scan() called but the class hasn't been setup properly.");

		return;
	}
	else if (m_size == 0)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScanDecoupledLookback::scan() called but no data uploaded.");

		return;
	}
	else if (!m_scan_kernel.has_been_compiled())
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelPrefixScanDecoupledLookback::scan() called but the kernels haven't been "
																		 "compiled. Did you forget to call ::compile()?");

		return;
	}
	else if (m_evenly_spaced_segment_size == 0 && m_flags_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "ParallelSegmentedReduction::scan() called but no segment information provided. "
																		 "Either upload segment flags or set an evenly spaced segment size.");

		Debug::debugbreak();

		return;
	}

	if (m_evenly_spaced_segment_size == 0)
		m_segment_ids_prefix_scan.scan(false);

	T* input_buffer_pointer					 = m_input_data_pointer;
	unsigned int* flags_buffer_pointer		 = m_flags_data_pointer;
	unsigned int* segment_ids_buffer_pointer = m_segment_ids_prefix_scan.get_output_buffer().get_device_pointer();

	T* output_buffer_pointer = m_output_buffer.get_device_pointer();
	void* block_scan_args[]	 = { &input_buffer_pointer,			&flags_buffer_pointer,	&segment_ids_buffer_pointer,
								 &m_evenly_spaced_segment_size, &output_buffer_pointer, &m_size };
	m_scan_kernel.launch_asynchronous(PARALLEL_REDUCTION_CHUNK_SIZE, 1, m_size, 1, block_scan_args, m_stream);

	if (auto_stream_synchronize)
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_stream));
}

template <typename T>
float ParallelSegmentedReduction<T>::get_last_execution_time()
{
	m_scan_kernel.compute_execution_time();

	return m_scan_kernel.get_last_execution_time();
}

template <typename T>
constexpr std::string ParallelSegmentedReduction<T>::get_data_type_as_string() const

{
	if constexpr (std::is_same_v<T, unsigned int>)
		return "unsigned int";
	else if constexpr (std::is_same_v<T, float>)
		return "float";
	else
		static_assert(sizeof(T) == 0 /* forces failure */, "Unsupported data type for ParallelPrefixScanDecoupledLookback");
}

template <typename T>
OrochiBuffer<T>& ParallelSegmentedReduction<T>::get_output_buffer()
{
	return m_output_buffer;
}

template <typename T>
OrochiBuffer<T>& ParallelSegmentedReduction<T>::get_segment_ids_buffer()
{
	return m_segment_ids_prefix_scan.get_output_buffer();
}

template <typename T>
void ParallelSegmentedReduction<T>::unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	unit_test_basic(hiprt_ctx, stream);
	unit_test_data_type(hiprt_ctx, stream);
	unit_test_transform(hiprt_ctx, stream);
	unit_test_evenly_spaced_flags(hiprt_ctx, stream);
}

template <typename T>
void ParallelSegmentedReduction<T>::unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelSegmentedReduction<unsigned int> scanner(hiprt_ctx, stream);
	scanner.compile();

	unit_test_template(hiprt_ctx, stream, scanner);
}

template <typename T>
void ParallelSegmentedReduction<T>::unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelSegmentedReduction<float> scanner(hiprt_ctx, stream);
	scanner.compile();

	unit_test_template(hiprt_ctx, stream, scanner);
}

template <typename T>
void ParallelSegmentedReduction<T>::unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelSegmentedReduction<float> scanner(hiprt_ctx, stream);
	scanner.set_data_transform(std::make_unique<ComputeDataTransformMultiplyBy2>());
	scanner.compile();

	unit_test_template<float>(hiprt_ctx, stream, scanner, [](float val) { return val * 2; });
}

template <typename T>
void ParallelSegmentedReduction<T>::unit_test_evenly_spaced_flags(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelSegmentedReduction<float> scanner(hiprt_ctx, stream);
	scanner.set_evenly_spaced_segment_size(rand() % 50);
	scanner.compile();

	unit_test_template<float>(hiprt_ctx, stream, scanner);
}

template <typename T>
template <typename DataType>
void ParallelSegmentedReduction<T>::unit_test_template(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
													   oroStream_t stream,
													   ParallelSegmentedReduction<DataType>& scanner,
													   std::function<DataType(DataType&)> input_value_transform,
													   std::function<DataType(DataType&)> output_value_transform)
{
	std::mt19937 engine_uint(42);
	auto rng = std::bind(std::conditional_t<std::is_integral_v<DataType>, std::uniform_int_distribution<unsigned int>, std::uniform_real_distribution<float>>(
												 1, 5),
						 engine_uint);

	// Full tests with random sizes
	oroEvent_t scan_start;
	oroEvent_t scan_end;

	OROCHI_CHECK_ERROR(oroEventCreate(&scan_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&scan_end));

	for (int i = 0; i < 10; i++)
	{
		engine_uint.seed(i);

		unsigned int test_size = engine_uint() % 100000000 + 1;

		std::vector<DataType> input(test_size);
		std::vector<DataType> untransformed_input;
		std::vector<unsigned int> flags((test_size + 31) / 32, 0);

		unsigned int evenly_spaced_segment_size = scanner.get_evenly_spaced_segment_size();

		std::transform(input.begin(), input.end(), input.begin(), [&rng](DataType) { return rng(); });
		if (evenly_spaced_segment_size == 0)
			std::transform(flags.begin(), flags.end(), flags.begin(), [&engine_uint](unsigned int) { return engine_uint(); });
		else
		{
			for (size_t j = 0; j < test_size; j++)
			{
				if (j % evenly_spaced_segment_size == 0)
					flags[j / 32] |= (1u << (j % 32)); // Set the flag for the start of this segment
			}
		}

		untransformed_input = input;

		std::transform(input.begin(), input.end(), input.begin(), input_value_transform);

		// Adding 2% of random warps that are full zero
		if (evenly_spaced_segment_size == 0)
		{
			unsigned int warps_count = (test_size + 31) / 32;
			for (unsigned int w = 0; w < warps_count; w++)
				if (engine_uint() % 100 < 2) // 2% chance
					flags[w] = 0;			 // This warp will have all flags set to zero, meaning that all its elements belong to the same segment

			// Adding 1% of random blocks that are full zero
			unsigned int blocks_count = (test_size + PARALLEL_REDUCTION_CHUNK_SIZE - 1) / PARALLEL_REDUCTION_CHUNK_SIZE;
			for (unsigned int b = 0; b < blocks_count; b++)
			{
				if (engine_uint() % 100 < 1) // 1% chance
				{
					for (unsigned int w = 0; w < PARALLEL_REDUCTION_CHUNK_SIZE / 32; w++)
					{
						if ((b * (PARALLEL_REDUCTION_CHUNK_SIZE / 32) + w) < warps_count)
							// This block will have all flags set to zero, meaning that all its elements belong to the same segment
							flags[b * (PARALLEL_REDUCTION_CHUNK_SIZE / 32) + w] = 0;
					}
				}
			}
		}

		std::conditional_t<std::is_same_v<DataType, float>, float, DataType> running_sum = 0;
		std::vector<DataType> expected_output(test_size);

		auto start				= std::chrono::high_resolution_clock::now();
		unsigned int segment_id = 0;
		for (size_t j = 0; j < test_size; j++)
		{
			if (j > 0)
			{
				bool is_segment_start = flags[j / (sizeof(unsigned int) * 8)] & (1u << (j % (sizeof(unsigned int) * 8)));
				if (is_segment_start)
				{
					expected_output[segment_id++] = running_sum;

					running_sum = 0;
				}
				else
					expected_output[j] = 0;
			}

			running_sum += input[j];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms for " << test_size << " elements."
				  << std::endl;

		std::transform(expected_output.begin(), expected_output.end(), expected_output.begin(), output_value_transform);

		scanner.upload_input_data(untransformed_input, flags);

		OROCHI_CHECK_ERROR(oroEventRecord(scan_start, stream));
		unsigned int repeats = 1;
		for (int j = 0; j < repeats; j++)
		{
			scanner.reduce();
		}
		OROCHI_CHECK_ERROR(oroEventRecord(scan_end, stream));

		float elapsed_time_ms = 0.0f;
		OROCHI_CHECK_ERROR(oroEventSynchronize(scan_end));
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&elapsed_time_ms, scan_start, scan_end));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO,
								"\tParallelSegmentedReduction unit test %d: scanned %u elements in %.3f ms. %.3fGItems/s", i, test_size,
								elapsed_time_ms / repeats, test_size / (elapsed_time_ms * 1e6f / repeats));

		std::vector<DataType> output = scanner.get_output_buffer().download_data();

		// The output is packed into the first 'segment_id' entries and the rest is undefined so we're only checking those
		for (long long int j = 0; j < segment_id; j++)
		{
			double diff = hippt::abs((double)output[j] - (double)expected_output[j]);
			if (diff / output[j] * 100.0 > 0.01)
			{
				std::string formatter;
				if constexpr (std::is_integral_v<DataType>)
					formatter = "%u";
				else
					formatter = "%.8f";

				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
										("ParallelSegmentedReduction unit test failed for test %d at index %lld (size=%u): got " + formatter + ", expected " +
										 formatter)
																.c_str(),
										i, j, test_size, output[j], expected_output[j]);

				Debug::debugbreak();

				return;
			}
		}

		scanner.get_output_buffer().memset_whole_buffer(0);
	}

	OROCHI_CHECK_ERROR(oroEventDestroy(scan_start));
	OROCHI_CHECK_ERROR(oroEventDestroy(scan_end));
}
