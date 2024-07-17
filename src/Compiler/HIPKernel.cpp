/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/HIPKernelCompiler.h"
#include "Compiler/HIPKernel.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <hiprt/impl/Compiler.h>

HIPKernel::HIPKernel(const std::string& kernel_file_path, const std::string& kernel_function_name)
{
	m_kernel_file_path = kernel_file_path;
	m_kernel_function_name = kernel_function_name;
}

std::string HIPKernel::get_kernel_file_path()
{
	return m_kernel_file_path;
}

std::string HIPKernel::get_kernel_function_name()
{
	return m_kernel_function_name;
}

GPUKernelCompilerOptions& HIPKernel::get_compiler_options()
{
	return m_kernel_compiler_options;
}

void HIPKernel::set_kernel_file_path(const std::string& kernel_file_path)
{
	m_kernel_file_path = kernel_file_path;
}

void HIPKernel::set_kernel_function_name(const std::string& kernel_function_name)
{
	m_kernel_function_name = kernel_function_name;
}

void HIPKernel::set_compiler_options(const GPUKernelCompilerOptions& options)
{
	m_kernel_compiler_options = options;
}

void HIPKernel::compile(hiprtContext& hiprt_ctx)
{
	std::string cache_key;

	cache_key = HIPKernelCompiler::get_additional_cache_key(*this);
	m_kernel_function = HIPKernelCompiler::compile_kernel(*this, hiprt_ctx, true, cache_key);

	//std::cout << "Registers: " << get_kernel_attribute(m_device_properties, ORO_FUNC_ATTRIBUTE_NUM_REGS) << std::endl;
	//std::cout << "Shared memory: " << get_kernel_attribute(m_device_properties, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) << std::endl;
}

int HIPKernel::get_kernel_attribute(oroDeviceProp device_properties, oroFunction_attribute attribute)
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

void HIPKernel::launch(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args)
{
	if (!m_events_created)
	{
		// Creating the events for later use
		OROCHI_CHECK_ERROR(oroEventCreate(&m_execution_start_event));
		OROCHI_CHECK_ERROR(oroEventCreate(&m_execution_stop_event));

		m_events_created = true;
	}

	hiprtInt2 nb_groups;
	nb_groups.x = std::ceil(static_cast<float>(res_x) / tile_size_x);
	nb_groups.y = std::ceil(static_cast<float>(res_y) / tile_size_y);

	OROCHI_CHECK_ERROR(oroModuleLaunchKernel(m_kernel_function, nb_groups.x, nb_groups.y, 1, tile_size_x, tile_size_y, 1, 0, 0, launch_args, 0));
}

void HIPKernel::launch_timed(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, float* execution_time_out)
{
	if (!m_events_created)
	{
		// Creating the events for later use
		OROCHI_CHECK_ERROR(oroEventCreate(&m_execution_start_event));
		OROCHI_CHECK_ERROR(oroEventCreate(&m_execution_stop_event));

		m_events_created = true;
	}

	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_start_event, 0));

	launch(tile_size_x, tile_size_y, res_x, res_y, launch_args);

	OROCHI_CHECK_ERROR(oroEventRecord(m_execution_stop_event, 0));
	OROCHI_CHECK_ERROR(oroEventSynchronize(m_execution_stop_event));
	OROCHI_CHECK_ERROR(oroEventElapsedTime(execution_time_out, m_execution_start_event, m_execution_stop_event));
}
