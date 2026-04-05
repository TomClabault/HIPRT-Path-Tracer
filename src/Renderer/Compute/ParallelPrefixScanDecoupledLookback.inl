/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Renderer/Compute/DataTransforms/ComputeDataTransforms.h"
#include "Renderer/Compute/ParallelPrefixScanDecoupledLookback.h"

#include <functional>
#include <random>

template <typename InputType, typename TransformedType, typename OutputType>
ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::ParallelPrefixScanDecoupledLookback()
	: m_hiprt_ctx(nullptr), m_stream(nullptr), m_size(0)
{
}

template <typename InputType, typename TransformedType, typename OutputType>
ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::ParallelPrefixScanDecoupledLookback(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
																												 oroStream_t stream)
	: m_size(0)
{
	init(hiprt_ctx, stream);
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::init(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	m_hiprt_ctx = hiprt_ctx;
	m_stream	= stream;

	initialize_kernels();
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::initialize_kernels()
{
	m_scan_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelPrefixScanDecoupledLookback/Scan.h");
	m_scan_kernel.set_kernel_function_name("ParallelPrefixScanDecoupledLookback_Scan");
	m_scan_kernel.get_kernel_options().set_string_macro_value("InputDataType", get_input_data_type_as_string());
	m_scan_kernel.get_kernel_options().set_string_macro_value("TransformedDataType", get_transformed_data_type_as_string());
	m_scan_kernel.get_kernel_options().set_string_macro_value("OutputDataType", get_output_data_type_as_string());

	m_scan_kernel.set_measure_execution_time(false);

	m_block_descriptor_init_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Compute/ParallelPrefixScanDecoupledLookback/BlockDescriptorInit.h");
	m_block_descriptor_init_kernel.set_kernel_function_name("ParallelPrefixScanDecoupledLookback_BlockDescriptorInit");
	m_block_descriptor_init_kernel.get_kernel_options().set_string_macro_value("InputDataType", get_input_data_type_as_string());
	m_block_descriptor_init_kernel.get_kernel_options().set_string_macro_value("TransformedDataType", get_transformed_data_type_as_string());
	m_block_descriptor_init_kernel.get_kernel_options().set_string_macro_value("OutputDataType", get_output_data_type_as_string());
	m_block_descriptor_init_kernel.set_measure_execution_time(false);

	set_input_id_transform(std::make_unique<ComputeInputIDTransformIdentity>());
	set_data_transform(std::make_unique<ComputeDataTransformIdentity>());
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::set_input_id_transform(std::unique_ptr<ComputeInputIDTransform> id_transform)
{
	m_scan_kernel.get_kernel_options().set_string_macro_value(ComputeInputIDTransform::INPUT_ID_TRANSFORM_STRING_STUB, id_transform->emit_input_id_transform());
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::set_data_transform(std::unique_ptr<ComputeDataTransform> transform)
{
	m_scan_kernel.get_kernel_options().set_string_macro_value(ComputeDataTransform::INPUT_DATA_TRANSFORM_STRING_STUB, transform->emit_input_transform());
	m_scan_kernel.get_kernel_options().set_string_macro_value(ComputeDataTransform::OUTPUT_DATA_TRANSFORM_STRING_STUB, transform->emit_output_transform());
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::set_exclusive_scan(bool exclusive)
{
	m_exclusive_scan = exclusive;
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::compile()
{
	m_scan_kernel.compile(m_hiprt_ctx, {}, true, false);
	m_block_descriptor_init_kernel.compile(m_hiprt_ctx, {}, true, false);
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::resize(unsigned int element_count)
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

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::free()
{
	m_input_buffer.free_no_error();
	m_output_buffer.free_no_error();
	m_global_block_index_counter_buffer.free_no_error();
	m_block_descriptors_buffer.free_no_error();

	m_input_data_pointer  = nullptr;
	m_output_data_pointer = nullptr;

	m_size						= 0;
	m_last_resize_element_count = 0;
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::upload_input_data(const std::vector<InputType>& data)
{
	m_size = data.size();

	resize(m_size);

	m_input_buffer.upload_data(data);

	m_input_data_pointer  = m_input_buffer.get_device_pointer();
	m_output_data_pointer = m_output_buffer.get_device_pointer();
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::set_data_pointers(InputType* input_buffer_pointer, unsigned int element_count)
{
	set_data_pointers(input_buffer_pointer, nullptr, element_count);
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::set_data_pointers(InputType* input_buffer_pointer,
																									OutputType* output_buffer_pointer,
																									unsigned int element_count)
{
	if (m_last_resize_element_count < element_count)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
								"ParallelPrefixScanDecoupledLookback::set_data_pointers() called with an element_count (%u) that is different from the last "
								"one used in resize() (%u). This is invalid usage.",
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

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::scan(bool auto_stream_synchronize)
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

	ParallelPrefixScanDecoupledLookbackBlockDescriptor* block_descriptors_buffer_pointer = m_block_descriptors_buffer.get_device_pointer();

	unsigned int block_descriptor_count				 = m_block_descriptors_buffer.size();
	unsigned int* global_block_index_counter_pointer = m_global_block_index_counter_buffer.get_device_pointer();
	void* block_descriptor_init_args[]				 = { &block_descriptors_buffer_pointer, &block_descriptor_count, &global_block_index_counter_pointer };
	m_block_descriptor_init_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, block_descriptor_count, 1, block_descriptor_init_args, m_stream);

	InputType* input_buffer_pointer	  = m_input_data_pointer;
	OutputType* output_buffer_pointer = m_output_data_pointer;
	void* block_scan_args[] = { &input_buffer_pointer, &output_buffer_pointer, &block_descriptors_buffer_pointer, &global_block_index_counter_pointer, &m_size,
								&m_exclusive_scan };
	m_scan_kernel.launch_asynchronous(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, 1, m_size, 1, block_scan_args, m_stream);

	if (auto_stream_synchronize)
		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_stream));
}

template <typename InputType, typename TransformedType, typename OutputType>
float ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::get_last_execution_time()
{
	m_block_descriptor_init_kernel.compute_execution_time();
	m_scan_kernel.compute_execution_time();

	return m_block_descriptor_init_kernel.get_last_execution_time() + m_scan_kernel.get_last_execution_time();
}

template <typename InputType, typename TransformedType, typename OutputType>
constexpr std::string ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::get_input_data_type_as_string() const
{
	if constexpr (std::is_same_v<InputType, unsigned int>)
		return "unsigned int";
	else if constexpr (std::is_same_v<InputType, float>)
		return "float";
	else
		static_assert(sizeof(InputType) == 0 /* forces failure */, "Unsupported data type for ParallelPrefixScanDecoupledLookback");
}

template <typename InputType, typename TransformedType, typename OutputType>
constexpr std::string ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::get_transformed_data_type_as_string() const
{
	if constexpr (std::is_same_v<TransformedType, unsigned int>)
		return "unsigned int";
	else if constexpr (std::is_same_v<TransformedType, float>)
		return "float";
	else
		static_assert(sizeof(TransformedType) == 0 /* forces failure */, "Unsupported data type for ParallelPrefixScanDecoupledLookback");
}

template <typename InputType, typename TransformedType, typename OutputType>
constexpr std::string ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::get_output_data_type_as_string() const
{
	if constexpr (std::is_same_v<OutputType, unsigned int>)
		return "unsigned int";
	else if constexpr (std::is_same_v<OutputType, float>)
		return "float";
	else
		static_assert(sizeof(OutputType) == 0 /* forces failure */, "Unsupported data type for ParallelPrefixScanDecoupledLookback");
}

template <typename InputType, typename TransformedType, typename OutputType>
OrochiBuffer<OutputType>& ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::get_output_buffer()
{
	return m_output_buffer;
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	unit_test_basic(hiprt_ctx, stream);
	unit_test_data_type(hiprt_ctx, stream);
	unit_test_data_type_uint_to_float(hiprt_ctx, stream);
	unit_test_transform(hiprt_ctx, stream);
	unit_test_inclusive(hiprt_ctx, stream);
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test_basic(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, oroStream_t stream)
{
	ParallelPrefixScanDecoupledLookback<unsigned int> scanner(hiprt_ctx, stream);
	scanner.compile();

	unit_test_template(hiprt_ctx, stream, scanner);
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test_data_type(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
																									  oroStream_t stream)
{
	ParallelPrefixScanDecoupledLookback<float> scanner(hiprt_ctx, stream);
	scanner.compile();

	unit_test_template(hiprt_ctx, stream, scanner);
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test_data_type_uint_to_float(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
																													oroStream_t stream)
{
	class ComputeDataTransformUintToFloat : public ComputeDataTransform
	{
	public:
		virtual std::string emit_input_transform() const override
		{
			// Some random transform to take alternating bits: 0b1010101010101010 = 0xAAAA
			return "return (float)(value & 0xAAAAAAAA);";
		}

		virtual std::string emit_output_transform() const override
		{
			return "return value;";
		}
	};

	ParallelPrefixScanDecoupledLookback<unsigned int, float, float> scanner(hiprt_ctx, stream);
	scanner.set_data_transform(std::make_unique<ComputeDataTransformUintToFloat>());
	scanner.compile();

	unit_test_template<unsigned int, float, float>(hiprt_ctx, stream, scanner, [](unsigned int val) { return (float)(val & 0xAAAAAAAA); });
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test_transform(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
																									  oroStream_t stream)
{
	ParallelPrefixScanDecoupledLookback<float> scanner(hiprt_ctx, stream);
	scanner.set_data_transform(std::make_unique<ComputeDataTransformMultiplyBy2>());
	scanner.compile();

	unit_test_template<float, float, float>(hiprt_ctx, stream, scanner, [](float val) { return val * 2; });
}

template <typename InputType, typename TransformedType, typename OutputType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test_inclusive(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
																									  oroStream_t stream)
{
	ParallelPrefixScanDecoupledLookback<unsigned int> scanner(hiprt_ctx, stream);
	scanner.set_exclusive_scan(false);
	scanner.compile();

	unit_test_template<unsigned int, unsigned int, unsigned int>(
							hiprt_ctx, stream, scanner, [](unsigned int val) { return val; }, [](unsigned int val) { return val; }, false);
}

template <typename InputType, typename TransformedType, typename OutputType>
template <typename InputDataType, typename TransformedDataType, typename OutputDataType>
void ParallelPrefixScanDecoupledLookback<InputType, TransformedType, OutputType>::unit_test_template(
						std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx,
						oroStream_t stream,
						ParallelPrefixScanDecoupledLookback<InputDataType, TransformedDataType, OutputDataType>& scanner,
						std::function<TransformedDataType(InputDataType&)> input_value_transform,
						std::function<OutputDataType(TransformedDataType&)> output_value_transform,
						bool exclusive_scan)
{
	std::mt19937 engine_uint(42);
	auto rng = std::bind(std::conditional_t<std::is_integral_v<InputDataType>, std::uniform_int_distribution<unsigned int>,
											std::uniform_real_distribution<float>>(0, 100),
						 engine_uint);

	// Full tests with random sizes
	oroEvent_t scan_start;
	oroEvent_t scan_end;

	OROCHI_CHECK_ERROR(oroEventCreate(&scan_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&scan_end));

	for (int i = 0; i < 10; i++)
	{
		engine_uint.seed(i);

		unsigned int test_size = engine_uint() % 10000000 + 1;

		std::vector<InputDataType> input(test_size);
		std::vector<OutputDataType> expected_output(test_size);

		std::transform(input.begin(), input.end(), input.begin(), [&rng](InputDataType) { return rng(); });
		std::vector<TransformedDataType> transformed_input(input.size());
		// Transform input for CPU computation
		std::transform(input.begin(), input.end(), transformed_input.begin(), input_value_transform);

		auto start																					  = std::chrono::high_resolution_clock::now();
		std::conditional_t<std::is_same_v<OutputDataType, float>, double, OutputDataType> running_sum = 0;
		for (size_t j = 0; j < test_size; j++)
		{
			if (exclusive_scan)
			{
				expected_output[j] = running_sum;
				running_sum += transformed_input[j];
			}
			else
			{
				running_sum += transformed_input[j];
				expected_output[j] = running_sum;
			}
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms for " << test_size << " elements."
				  << std::endl;

		std::transform(expected_output.begin(), expected_output.end(), expected_output.begin(), output_value_transform);

		// Upload untransformed input
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

		std::vector<OutputDataType> output = scanner.get_output_buffer().download_data();

		for (long long int j = 0; j < test_size; j++)
		{
			double diff = hippt::abs((double)output[j] - (double)expected_output[j]);
			if (diff / output[j] * 100.0 > 0.001)
			{
				// More than a certain percentage of error

				std::string formatter;
				if constexpr (std::is_integral_v<OutputDataType>)
					formatter = "%u";
				else
					formatter = "%f";

				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
										("ParallelPrefixScanDecoupledLookback unit test failed for test %d at index %lld (size=%u): got " + formatter +
										 ", expected " + formatter)
																.c_str(),
										i, j, test_size, output[j], expected_output[j]);

				Debug::debugbreak();

				return;
			}
		}
	}

	OROCHI_CHECK_ERROR(oroEventDestroy(scan_start));
	OROCHI_CHECK_ERROR(oroEventDestroy(scan_end));
}
