/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "Compiler/GPUKernel.h"
#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/ImGui/ImGuiLogger.h"

extern GPUKernelCompiler g_gpu_kernel_compiler;
extern ImGuiLogger g_imgui_logger;

const std::vector<std::string> GPUKernel::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS =
{
	KERNEL_COMPILER_ADDITIONAL_INCLUDE,
	DEVICE_INCLUDES_DIRECTORY,
	OROCHI_INCLUDES_DIRECTORY,
	"./"
};

GPUKernel::GPUKernel()
{
	OROCHI_CHECK_ERROR(oroEventCreate(&m_execution_start_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_execution_stop_event));
}

GPUKernel::GPUKernel(const std::string& kernel_file_path, const std::string& kernel_function_name) : GPUKernel()
{
	m_kernel_file_path = kernel_file_path;
	m_kernel_function_name = kernel_function_name;
}

std::string GPUKernel::get_kernel_file_path() const
{
	return m_kernel_file_path;
}

std::string GPUKernel::get_kernel_function_name() const
{
	return m_kernel_function_name;
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
	if (m_option_macro_invalidated)
		parse_option_macros_used();

	std::string cache_key = g_gpu_kernel_compiler.get_additional_cache_key(*this);
	m_kernel_function = g_gpu_kernel_compiler.compile_kernel(*this, m_compiler_options, hiprt_ctx,
															 func_name_sets.data(), 
															 /* num geom */1,
															 /* num ray */ func_name_sets.size() == 0 ? 0 : 1,
															 use_cache, cache_key, silent);
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
		int macro_value = *macro_to_value.second;

		if (options_excluded.find(macro_name) == options_excluded.end())
			// Option is not excluded
			m_compiler_options.set_pointer_to_macro(macro_name, other_options->get_pointer_to_macro_value(macro_name));
	}

	// Same thing with the custom macros
	for (auto macro_to_value : other_options->get_custom_macro_map())
	{
		const std::string& macro_name = macro_to_value.first;
		int macro_value = *macro_to_value.second;

		if (options_excluded.find(macro_name) == options_excluded.end())
			// Option is not excluded
			m_compiler_options.set_pointer_to_macro(macro_name, other_options->get_pointer_to_macro_value(macro_name));
	}
}

void GPUKernel::launch(int block_size_x, int block_size_y, int nb_threads_x, int nb_threads_y, void** launch_args, oroStream_t stream)
{
	launch_3D(block_size_x, block_size_y, 1, nb_threads_x, nb_threads_y, 1, launch_args, stream);
}

void GPUKernel::launch_3D(int block_size_x, int block_size_y, int block_size_z, int nb_threads_x, int nb_threads_y, int nb_threads_z, void** launch_args, oroStream_t stream)
{
	int3 nb_groups;
	nb_groups.x = std::ceil(static_cast<float>(nb_threads_x) / block_size_x);
	nb_groups.y = std::ceil(static_cast<float>(nb_threads_y) / block_size_y);
	nb_groups.z = std::ceil(static_cast<float>(nb_threads_z) / block_size_z);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_kernel_function, nb_groups.x, nb_groups.y, nb_groups.z, block_size_x, block_size_y, block_size_z, 0, stream, launch_args, 0));
	m_launched_at_least_once = true;
}

void GPUKernel::launch_synchronous(int block_size_x, int block_size_y, int nb_threads_x, int nb_threads_y, void** launch_args, float* execution_time_out)
{
	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_start_event, 0));

	launch(block_size_x, block_size_y, nb_threads_x, nb_threads_y, launch_args, 0);

	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_stop_event, 0));
	OROCHI_CHECK_ERROR(oroEventSynchronize(m_execution_stop_event));
	if (execution_time_out != nullptr)
		OROCHI_CHECK_ERROR(oroEventElapsedTime(execution_time_out, m_execution_start_event, m_execution_stop_event));
}

void GPUKernel::parse_option_macros_used()
{
	m_used_option_macros = g_gpu_kernel_compiler.get_option_macros_used_by_kernel(*this);
	m_option_macro_invalidated = false;
}

bool GPUKernel::uses_macro(const std::string& name) const
{
	return m_used_option_macros.find(name) != m_used_option_macros.end();
}

float GPUKernel::get_last_execution_time()
{
	if (!m_launched_at_least_once)
		return 0.0f;

	float out;
	OROCHI_CHECK_ERROR(oroEventElapsedTime(&out, m_execution_start_event, m_execution_stop_event));

	return out;
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

void GPUKernel::launch_asynchronous(int block_size_x, int block_size_y, int nb_threads_x, int nb_threads_y, void** launch_args, oroStream_t stream)
{
	launch_asynchronous_3D(block_size_x, block_size_y, 1, nb_threads_x, nb_threads_y, 1, launch_args, stream);
}

void GPUKernel::launch_asynchronous_3D(int block_size_x, int block_size_y, int block_size_z, int nb_threads_x, int nb_threads_y, int nb_threads_z, void** launch_args, oroStream_t stream)
{
	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_start_event, stream));

	launch_3D(block_size_x, block_size_y, block_size_z, nb_threads_x, nb_threads_y, nb_threads_z, launch_args, stream);

	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_stop_event, stream));

	// TODO: There's an issue here on HIP 5.7 + Windows where without the oroLaunchHostFunc below,
	// this oroEventRecord (or any event after a kernel launch) "blocks" the stream (only on a non-NULL stream)
	// and oroStreamQuery always (kind of) returns hipErrorDeviceNotReady
	oroLaunchHostFunc(stream, [](void*) {}, nullptr);
}
