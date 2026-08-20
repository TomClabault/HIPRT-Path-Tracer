/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernel.h"
#include "Compiler/GPUKernelCompiler.h"
#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/ImGui/ImGuiLogger.h"

extern GPUKernelCompiler g_gpu_kernel_compiler;
extern ImGuiLogger g_imgui_logger;

const std::vector<std::string> GPUKernel::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS = { KERNEL_COMPILER_ADDITIONAL_INCLUDE, DEVICE_INCLUDES_DIRECTORY,
																					OROCHI_INCLUDES_DIRECTORY, "./" };

GPUKernel::GPUKernel() = default;

GPUKernel::GPUKernel(const std::string& kernel_name) : GPUKernel()
{
	m_name = kernel_name;
}

GPUKernel::GPUKernel(const std::string& kernel_file_path, const std::string& kernel_function_name) : GPUKernel()
{
	m_kernel_file_path	   = kernel_file_path;
	m_kernel_function_name = kernel_function_name;
}

std::string GPUKernel::get_kernel_name() const
{
	if (m_name != "")
		return m_name;
	else
		return m_kernel_function_name;
}

std::string GPUKernel::get_kernel_file_path() const
{
	return m_kernel_file_path;
}

std::string GPUKernel::get_kernel_function_name() const
{
	return m_kernel_function_name;
}

void GPUKernel::set_kernel_name(const std::string& kernel_name)
{
	m_name = kernel_name;
}

void GPUKernel::set_kernel_file_path(const std::string& kernel_file_path)
{
	m_kernel_file_path = kernel_file_path;
}

void GPUKernel::set_kernel_function_name(const std::string& kernel_function_name)
{
	m_kernel_function_name = kernel_function_name;
}

void GPUKernel::add_additional_macro_for_compilation(const std::string& name, int value)
{
	m_additional_compilation_macros[name] = value;
}

std::vector<std::string> GPUKernel::get_additional_compiler_macros() const
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_additional_compilation_macros)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	return macros;
}

void GPUKernel::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, std::vector<hiprtFuncNameSet> func_name_sets, bool use_cache, bool silent)
{
	m_module_globals_cache.clear();

	if (m_option_macro_invalidated)
		parse_option_macros_used();

	std::string cache_key = g_gpu_kernel_compiler.get_additional_cache_key(*this);

	m_kernel_module	  = nullptr;
	m_kernel_function = g_gpu_kernel_compiler.compile_kernel(*this, m_compiler_options, hiprt_ctx, func_name_sets.data(),
															 /* num geom */ 1,
															 /* num ray */ func_name_sets.size() == 0 ? 0 : 1, use_cache, cache_key, silent, &m_kernel_module);
}

void GPUKernel::upload_to_module_global(const char* global_name, const void* data, size_t data_size, oroStream_t stream)
{
	if (m_kernel_module == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to upload to a module global before the kernel was compiled.");

		return;
	}

	std::string global_name_string = global_name;

	auto module_global_iterator = m_module_globals_cache.find(global_name_string);
	if (module_global_iterator == m_module_globals_cache.end())
	{
		ModuleGlobal module_global{ nullptr, 0 };

		OROCHI_CHECK_ERROR(oroModuleGetGlobal(&module_global.device_pointer, &module_global.size, m_kernel_module, global_name));

		// Cache the module for the next time we want to upload to this global
		module_global_iterator = m_module_globals_cache.emplace(global_name_string, module_global).first;
	}

	if (module_global_iterator->second.size != data_size)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Module global '%s' has size %zu, but %zu bytes were provided.", global_name,
								module_global_iterator->second.size, data_size);

		return;
	}

	OROCHI_CHECK_ERROR(oroMemcpyAsync(reinterpret_cast<void*>(module_global_iterator->second.device_pointer), data, data_size, oroMemcpyHostToDevice, stream));
}

int GPUKernel::get_kernel_attribute(oroFunction compiled_kernel, oroFunction_attribute attribute)
{
	int numRegs = 0;

	if (compiled_kernel == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to get an attribute of a kernel that wasn't compiled yet.");

		return 0;
	}

	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, attribute, compiled_kernel));

	return numRegs;
}

int GPUKernel::get_kernel_attribute(oroFunction_attribute attribute) const
{
	int numRegs = 0;

	if (m_kernel_function == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to get an attribute of a kernel that wasn't compiled yet.");

		return 0;
	}

	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, attribute, m_kernel_function));

	return numRegs;
}

GPUKernelCompilerOptions& GPUKernel::get_kernel_options()
{
	return m_compiler_options;
}

const GPUKernelCompilerOptions& GPUKernel::get_kernel_options() const
{
	return m_compiler_options;
}

void GPUKernel::synchronize_options_with(std::shared_ptr<GPUKernelCompilerOptions> other_options, const std::unordered_set<std::string>& options_excluded)
{
	for (auto macro_to_value : other_options->get_options_macro_map())
	{
		const std::string& macro_name = macro_to_value.first;
		int macro_value				  = *macro_to_value.second;

		if (options_excluded.find(macro_name) == options_excluded.end())
			// Option is not excluded
			m_compiler_options.set_pointer_to_macro(macro_name, other_options->get_pointer_to_macro_value(macro_name));
	}

	// Same thing with the custom macros
	for (auto macro_to_value : other_options->get_custom_macro_map())
	{
		const std::string& macro_name = macro_to_value.first;
		int macro_value				  = *macro_to_value.second;

		if (options_excluded.find(macro_name) == options_excluded.end())
			// Option is not excluded
			m_compiler_options.set_pointer_to_macro(macro_name, other_options->get_pointer_to_macro_value(macro_name));
	}
}

void GPUKernel::launch(int block_size_x, int block_size_y, int nb_threads_x, int nb_threads_y, void** launch_args, oroStream_t stream)
{
	launch_3D_block_size(block_size_x, block_size_y, 1, nb_threads_x, nb_threads_y, 1, launch_args, stream);
}

void GPUKernel::launch_3D_block_size(
	int block_size_x, int block_size_y, int block_size_z, int nb_threads_x, int nb_threads_y, int nb_threads_z, void** launch_args, oroStream_t stream)
{
	unsigned int block_count_x = (nb_threads_x + block_size_x - 1) / block_size_x;
	unsigned int block_count_y = (nb_threads_y + block_size_y - 1) / block_size_y;
	unsigned int block_count_z = (nb_threads_z + block_size_z - 1) / block_size_z;

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_kernel_function, block_count_x, block_count_y, block_count_z, block_size_x, block_size_y, block_size_z, 0,
											 stream, launch_args, 0));
}

void GPUKernel::launch_asynchronous_3D_block_count(
	int block_count_x, int block_count_y, int block_count_z, int block_size_x, int block_size_y, int block_size_z, void** launch_args, oroStream_t stream)
{
	if (m_measure_execution_time)
		record_execution_start(stream);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_kernel_function, block_count_x, block_count_y, block_count_z, block_size_x, block_size_y, block_size_z, 0,
											 stream, launch_args, 0));

	if (m_measure_execution_time)
		record_execution_stop(stream);
}

void GPUKernel::launch_synchronous(int block_size_x, int block_size_y, int nb_threads_x, int nb_threads_y, void** launch_args, float* execution_time_out)
{
	if (!m_measure_execution_time)
	{
		launch(block_size_x, block_size_y, nb_threads_x, nb_threads_y, launch_args, 0);

		if (execution_time_out != nullptr)
			*execution_time_out = 0.0f;

		return;
	}

	record_execution_start(0);

	launch(block_size_x, block_size_y, nb_threads_x, nb_threads_y, launch_args, 0);

	record_execution_stop(0);

	std::size_t execution_event_index = m_recorded_execution_event_count - 1;
	OROCHI_CHECK_ERROR(oroEventSynchronize(m_execution_stop_events[execution_event_index]));
	if (execution_time_out != nullptr)
		OROCHI_CHECK_ERROR(
			oroEventElapsedTime(execution_time_out, m_execution_start_events[execution_event_index], m_execution_stop_events[execution_event_index]));
}

void GPUKernel::record_execution_start(oroStream_t stream)
{
	if (m_recorded_execution_event_count == m_execution_start_events.size())
	{
		oroEvent_t start_event = nullptr;
		oroEvent_t stop_event  = nullptr;

		OROCHI_CHECK_ERROR(oroEventCreate(&start_event));
		OROCHI_CHECK_ERROR(oroEventCreate(&stop_event));

		m_execution_start_events.push_back(start_event);
		m_execution_stop_events.push_back(stop_event);
	}

	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_start_events[m_recorded_execution_event_count], stream));
}

void GPUKernel::record_execution_stop(oroStream_t stream)
{
	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_stop_events[m_recorded_execution_event_count], stream));
	m_recorded_execution_event_count++;
}

void GPUKernel::parse_option_macros_used()
{
	m_used_option_macros	   = g_gpu_kernel_compiler.get_option_macros_used_by_kernel(*this);
	m_option_macro_invalidated = false;
}

bool GPUKernel::uses_macro(const std::string& name) const
{
	return m_used_option_macros.find(name) != m_used_option_macros.end();
}

float GPUKernel::compute_execution_time_and_reset_execution_count()
{
	if (!m_measure_execution_time)
	{
		m_recorded_execution_event_count = 0;
		m_last_execution_time			 = 0.0f;

		return 0.0f;
	}

	float total_execution_time = 0.0f;
	for (std::size_t event_index = 0; event_index < m_recorded_execution_event_count; event_index++)
	{
		float execution_time = 0.0f;
		OROCHI_CHECK_ERROR(oroEventElapsedTime(&execution_time, m_execution_start_events[event_index], m_execution_stop_events[event_index]));

		total_execution_time += execution_time;
	}

	m_last_execution_time			 = total_execution_time;
	m_recorded_execution_event_count = 0;

	return total_execution_time;
}

float GPUKernel::get_last_execution_time() const
{
	if (!m_measure_execution_time)
		return 0.0f;

	return m_last_execution_time;
}

bool GPUKernel::has_been_compiled() const
{
	return m_kernel_function != nullptr;
}

bool GPUKernel::is_precompiled() const
{
	return m_is_precompiled_kernel;
}

void GPUKernel::set_precompiled(bool precompiled)
{
	m_is_precompiled_kernel = precompiled;
}

bool GPUKernel::is_measuring_execution_time() const
{
	return m_measure_execution_time;
}

void GPUKernel::set_measure_execution_time(bool measure_execution_time)
{
	m_measure_execution_time = measure_execution_time;

	if (!measure_execution_time)
	{
		m_recorded_execution_event_count = 0;
		m_last_execution_time			 = 0.0f;
	}
}

void GPUKernel::launch_asynchronous(int block_size_x, int block_size_y, int nb_threads_x, int nb_threads_y, void** launch_args, oroStream_t stream)
{
	launch_asynchronous_3D(block_size_x, block_size_y, 1, nb_threads_x, nb_threads_y, 1, launch_args, stream);
}

void GPUKernel::launch_asynchronous_3D(
	int block_size_x, int block_size_y, int block_size_z, int nb_threads_x, int nb_threads_y, int nb_threads_z, void** launch_args, oroStream_t stream)
{
	if (m_measure_execution_time)
		record_execution_start(stream);

	launch_3D_block_size(block_size_x, block_size_y, block_size_z, nb_threads_x, nb_threads_y, nb_threads_z, launch_args, stream);

	if (m_measure_execution_time)
		record_execution_stop(stream);
}
