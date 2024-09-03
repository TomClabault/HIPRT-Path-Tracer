/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "Compiler/GPUKernel.h"
#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <hiprt/impl/Compiler.h>

extern GPUKernelCompiler g_gpu_kernel_compiler;

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

void GPUKernel::compile(hiprtContext& hiprt_ctx, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, bool use_cache)
{
	if (!m_option_macro_parsed)
		parse_option_macros_used(kernel_compiler_options);

	std::string cache_key = g_gpu_kernel_compiler.get_additional_cache_key(*this, kernel_compiler_options);
	m_kernel_function = g_gpu_kernel_compiler.compile_kernel(*this, kernel_compiler_options, hiprt_ctx, use_cache, cache_key);
}

int GPUKernel::get_kernel_attribute(oroFunction compiled_kernel, oroFunction_attribute attribute)
{
	int numRegs = 0;

	if (compiled_kernel == nullptr)
	{
		std::cerr << "Trying to get an attribute of a kernel that wasn't compiled yet." << std::endl;

		return 0;
	}

	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, compiled_kernel));

	return numRegs;
}

int GPUKernel::get_kernel_attribute(oroFunction_attribute attribute)
{
	int numRegs = 0;

	if (m_kernel_function == nullptr)
	{
		std::cerr << "Trying to get an attribute of a kernel that wasn't compiled yet." << std::endl;

		return 0;
	}

	OROCHI_CHECK_ERROR(oroFuncGetAttribute(&numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, m_kernel_function));

	return numRegs;
}

void GPUKernel::launch(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream)
{
	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(static_cast<float>(res_x) / tile_size_x);
	nb_groups.y = std::ceil(static_cast<float>(res_y) / tile_size_y);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_kernel_function, nb_groups.x, nb_groups.y, 1, tile_size_x, tile_size_y, 1, 0, stream, launch_args, 0));
}

void GPUKernel::launch_timed_synchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, float* execution_time_out)
{
	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_start_event, 0));

	launch(tile_size_x, tile_size_y, res_x, res_y, launch_args, 0);

	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_stop_event, 0));
	OROCHI_CHECK_ERROR(oroEventSynchronize(m_execution_stop_event));
	OROCHI_CHECK_ERROR(oroEventElapsedTime(execution_time_out, m_execution_start_event, m_execution_stop_event));
}

void GPUKernel::parse_option_macros_used(std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options)
{
	m_used_option_macros = g_gpu_kernel_compiler.get_option_macros_used_by_kernel(*this, kernel_compiler_options);
	m_option_macro_parsed = true;
}

bool GPUKernel::uses_macro(const std::string& name) const
{
	return m_used_option_macros.find(name) != m_used_option_macros.end();
}

void GPUKernel::compute_elapsed_time_callback(void* data)
{
	GPUKernel::ComputeElapsedTimeCallbackData* callback_data = reinterpret_cast<ComputeElapsedTimeCallbackData*>(data);
	oroEventElapsedTime(callback_data->elapsed_time_out, callback_data->start, callback_data->end);

	// Deleting the callback data because it was dynamically allocated
	delete callback_data;
}

float GPUKernel::get_last_execution_time()
{
	return m_last_execution_time;
}

void GPUKernel::launch_timed_asynchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream)
{
	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_start_event, stream));

	launch(tile_size_x, tile_size_y, res_x, res_y, launch_args, stream);

	// TODO: There's an issue here on HIP 5.7 + Windows where without the oroLaunchHostFunc below,
	// this oroEventRecord (or any event after a kernel launch) "blocks" the stream (only on a non-NULL stream)
	// and oroStreamQuery always (kind of) returns hipErrorDeviceNotReady
	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_stop_event, stream));

	GPUKernel::ComputeElapsedTimeCallbackData* callback_data = new ComputeElapsedTimeCallbackData;
	callback_data->elapsed_time_out = &m_last_execution_time;
	callback_data->start = m_execution_start_event;
	callback_data->end = m_execution_stop_event;

	// Automatically computing the elapsed time of the events with a callback.
	// hip/cudaLaunchHostFunc adds a host function call on the GPU queue. Pretty nifty
	OROCHI_CHECK_ERROR(oroLaunchHostFunc(stream, GPUKernel::compute_elapsed_time_callback, callback_data));
}
