/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/HIPKernel.h"
#include "HIPRT-Orochi/HIPKernelCompiler.h"
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

std::vector<std::string> HIPKernel::get_additional_include_directories()
{
	return m_additional_include_directories;
}

std::vector<std::string> HIPKernel::get_compiler_options()
{
	return m_compiler_options;
}

void HIPKernel::set_kernel_file_path(const std::string& kernel_file_path)
{
	m_kernel_file_path = kernel_file_path;
}

void HIPKernel::set_kernel_function_name(const std::string& kernel_function_name)
{
	m_kernel_function_name = kernel_function_name;
}

void HIPKernel::set_additional_includes(const std::vector<std::string>& additional_include_directories)
{
	m_additional_include_directories = additional_include_directories;
}

void HIPKernel::set_compiler_options(const std::vector<std::string>& compiler_options)
{
	m_compiler_options = compiler_options;
}

void HIPKernel::compile(hiprtContext& hiprt_ctx)
{
	std::string cache_key;

	cache_key = HIPKernelCompiler::get_additional_cache_key(*this);
	m_kernel_function = HIPKernelCompiler::compile_kernel(*this, hiprt_ctx, true, cache_key);
}

int HIPKernel::get_register_count(oroDeviceProp device_properties)
{
	int numRegs = 0;

	if (m_kernel_function == nullptr)
	{
		std::cerr << "Trying to get the number of registers used by a kernel that wasn't compiled yet." << std::endl;

		return 0;
	}

	if ((device_properties.major >= 7 && device_properties.minor >= 5) || std::string(device_properties.name).find("Radeon") != std::string::npos)
		// Getting the number of registers / shared memory of the function
		// doesn't work on a CC 5.2 NVIDIA GPU but does work on a 7.5 so
		// I'm assuming this is an issue of compute capability (although
		// it could be a completely different thing)
		//
		// Also this if() statement assumes that getting the number of
		// registers will work on any AMD card but this has only been
		// tested on a gfx1100 GPU. This should be tested on older hardware
		// if possible
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
