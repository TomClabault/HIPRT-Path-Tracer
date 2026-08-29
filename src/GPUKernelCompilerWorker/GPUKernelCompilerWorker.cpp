/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GPUKernelCompilerWorker/GPUKernelCompilerWorker.h"
#include "Compiler/GPUKernelCompilerWindowProcessSerialization.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max

bool GPUKernelCompilerWorker::initialize_worker_context(int device_index, hiprtContext& hiprt_context_out, oroCtx& orochi_context_out)
{
#ifdef OROCHI_ENABLE_CUEW
	int error_initialize = oroInitialize(static_cast<oroApi>(ORO_API_CUDA), 0);
#else // #ifdef OROCHI_ENABLE_CUEW
	int error_initialize = oroInitialize(static_cast<oroApi>(ORO_API_HIP), 0);
#endif // #ifdef OROCHI_ENABLE_CUEW
	if (error_initialize != oroSuccess || oroInit(0) != oroSuccess)
		return false;

	oroDevice orochi_device = 0;
	if (oroDeviceGet(&orochi_device, device_index) != oroSuccess || oroCtxCreate(&orochi_context_out, 0, orochi_device) != oroSuccess)
		return false;

	oroDeviceProp device_properties = {};
	if (oroGetDeviceProperties(&device_properties, orochi_device) != oroSuccess)
		return false;

	hiprtContextCreationInput context_input = { nullptr, -1, hiprtDeviceAMD };
	context_input.ctxt						= oroGetRawCtx(orochi_context_out);
	context_input.device					= oroGetRawDevice(orochi_device);
	if (std::string(device_properties.name).find("NVIDIA") != std::string::npos)
		context_input.deviceType = hiprtDeviceNVIDIA;

	if (hiprtCreateContext(HIPRT_API_VERSION, context_input, hiprt_context_out) != hiprtSuccess)
		return false;

	if (hiprtSetLogLevel(hiprt_context_out, hiprtLogLevelError) != hiprtSuccess)
		return false;

	return oroCtxSetCurrent(orochi_context_out) == oroSuccess;
}

void GPUKernelCompilerWorker::destroy_worker_context(hiprtContext hiprt_context, oroCtx orochi_context)
{
	if (hiprt_context != nullptr)
		hiprtDestroyContext(hiprt_context);
	if (orochi_context != nullptr)
		oroCtxDestroy(orochi_context);
}

bool GPUKernelCompilerWorker::compile_in_worker(const GPUKernelCompilerWindowProcessCompilationRequest& request)
{
	if (request.num_geom_types < 0 || request.num_ray_types < 0 || request.device_index < 0)
	{
		std::cout << "Invalid kernel compiler request dimensions or device index." << std::endl;
		return false;
	}

	std::vector<std::string> intersect_function_names;
	std::vector<std::string> filter_function_names;
	std::vector<hiprtFuncNameSet> function_name_sets;

	if (request.has_function_name_sets)
	{
		intersect_function_names.reserve(request.function_name_sets.size());
		filter_function_names.reserve(request.function_name_sets.size());
		function_name_sets.resize(request.function_name_sets.size());

		for (size_t function_name_set_index = 0; function_name_set_index < request.function_name_sets.size(); function_name_set_index++)
		{
			const GPUKernelCompilerWindowProcessFunctionNameSet& request_function_name_set = request.function_name_sets[function_name_set_index];
			hiprtFuncNameSet& function_name_set											   = function_name_sets[function_name_set_index];

			if (request_function_name_set.intersect_function_name.has_value())
			{
				intersect_function_names.push_back(request_function_name_set.intersect_function_name.value());
				function_name_set.intersectFuncName = intersect_function_names.back().c_str();
			}

			if (request_function_name_set.filter_function_name.has_value())
			{
				filter_function_names.push_back(request_function_name_set.filter_function_name.value());
				function_name_set.filterFuncName = filter_function_names.back().c_str();
			}
		}
	}

	hiprtContext hiprt_context = nullptr;
	oroCtx orochi_context	   = nullptr;
	if (!GPUKernelCompilerWorker::initialize_worker_context(request.device_index, hiprt_context, orochi_context))
	{
		std::cout << "Unable to initialize the worker HIPRT context." << std::endl;
		GPUKernelCompilerWorker::destroy_worker_context(hiprt_context, orochi_context);
		return false;
	}

	hiprtApiFunction kernel_function = nullptr;
	hiprtError compile_status		 = HIPPTOrochiUtils::build_trace_kernel(
		   hiprt_context, request.kernel_file_path, request.kernel_function_name, kernel_function, request.additional_include_directories,
		   request.compiler_options, static_cast<unsigned int>(request.num_geom_types), static_cast<unsigned int>(request.num_ray_types),
		   request.use_compiler_cache, request.has_function_name_sets ? function_name_sets.data() : nullptr, request.additional_cache_key, nullptr, false);

	GPUKernelCompilerWorker::destroy_worker_context(hiprt_context, orochi_context);
	if (compile_status != hiprtSuccess)
	{
		std::cout << "Unable to compile kernel '" << request.kernel_function_name << "' in the worker process." << std::endl;
		return false;
	}

	return true;
}

int GPUKernelCompilerWorker::run(const std::filesystem::path& request_file_path)
{
	try
	{
		GPUKernelCompilerWindowProcessCompilationRequest request;
		if (!GPUKernelCompilerWindowProcessSerialization::read_request(request_file_path, request))
		{
			std::cout << "Unable to read kernel compiler request file: " << request_file_path.string() << std::endl;
			return 1;
		}

		return GPUKernelCompilerWorker::compile_in_worker(request) ? 0 : 1;
	}
	catch (const std::exception& exception)
	{
		std::cout << "Kernel compiler worker failed: " << exception.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cout << "Kernel compiler worker failed with an unknown exception." << std::endl;
		return 1;
	}
}

#endif // _WIN32 // #ifdef _WIN32
