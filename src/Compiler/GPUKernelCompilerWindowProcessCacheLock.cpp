/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerWindowProcessCacheLock.h"

#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max

#include <string>

void GPUKernelCompilerWindowProcessCacheLock::update_request_hash(unsigned long long& request_hash, const void* data, size_t data_size)
{
	const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data);
	for (size_t byte_index = 0; byte_index < data_size; byte_index++)
	{
		request_hash ^= bytes[byte_index];
		request_hash *= 1099511628211ull;
	}
}

void GPUKernelCompilerWindowProcessCacheLock::update_request_hash(unsigned long long& request_hash, const std::string& value)
{
	unsigned int value_size = static_cast<unsigned int>(value.size());

	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, &value_size, sizeof(value_size));
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, value.data(), value.size());
}

void GPUKernelCompilerWindowProcessCacheLock::update_request_hash(unsigned long long& request_hash, int value)
{
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, &value, sizeof(value));
}

void GPUKernelCompilerWindowProcessCacheLock::update_request_hash(unsigned long long& request_hash, unsigned int value)
{
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, &value, sizeof(value));
}

void GPUKernelCompilerWindowProcessCacheLock::update_request_hash(unsigned long long& request_hash, bool value)
{
	unsigned int serialized_value = value ? 1 : 0;

	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, &serialized_value, sizeof(serialized_value));
}

unsigned long long GPUKernelCompilerWindowProcessCacheLock::get_request_hash(const GPUKernelCompilerWindowProcessCompilationRequest& request)
{
	unsigned long long request_hash = 1469598103934665603ull;

	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.kernel_file_path);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.kernel_function_name);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, static_cast<unsigned int>(request.additional_include_directories.size()));

	for (const std::string& include_directory : request.additional_include_directories)
		GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, include_directory);

	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, static_cast<unsigned int>(request.compiler_options.size()));

	for (const std::string& compiler_option : request.compiler_options)
		GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, compiler_option);

	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.num_geom_types);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.num_ray_types);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.use_compiler_cache);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.has_function_name_sets);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, static_cast<unsigned int>(request.function_name_sets.size()));

	for (const GPUKernelCompilerWindowProcessFunctionNameSet& function_name_set : request.function_name_sets)
	{
		GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, function_name_set.intersect_function_name.has_value());

		if (function_name_set.intersect_function_name.has_value())
			GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, function_name_set.intersect_function_name.value());

		GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, function_name_set.filter_function_name.has_value());

		if (function_name_set.filter_function_name.has_value())
			GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, function_name_set.filter_function_name.value());
	}

	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.additional_cache_key);
	GPUKernelCompilerWindowProcessCacheLock::update_request_hash(request_hash, request.device_index);

	return request_hash;
}

GPUKernelCompilerWindowProcessCacheLock::GPUKernelCompilerWindowProcessCacheLock(const GPUKernelCompilerWindowProcessCompilationRequest& request)
{
	std::wstring mutex_name = L"Local\\HIPRTPathTracerKernelCompile-" + std::to_wstring(GPUKernelCompilerWindowProcessCacheLock::get_request_hash(request));
	mutex_handle			= CreateMutexW(nullptr, FALSE, mutex_name.c_str());
	if (mutex_handle == nullptr)
		return;

	DWORD wait_result = WaitForSingleObject(reinterpret_cast<HANDLE>(mutex_handle), INFINITE);
	mutex_acquired	  = wait_result == WAIT_OBJECT_0 || wait_result == WAIT_ABANDONED;
}

GPUKernelCompilerWindowProcessCacheLock::~GPUKernelCompilerWindowProcessCacheLock()
{
	if (mutex_acquired)
		ReleaseMutex(reinterpret_cast<HANDLE>(mutex_handle));
	if (mutex_handle != nullptr)
		CloseHandle(reinterpret_cast<HANDLE>(mutex_handle));
}

bool GPUKernelCompilerWindowProcessCacheLock::acquired() const
{
	return mutex_acquired;
}

#else

GPUKernelCompilerWindowProcessCacheLock::GPUKernelCompilerWindowProcessCacheLock(const GPUKernelCompilerWindowProcessCompilationRequest&) {}

GPUKernelCompilerWindowProcessCacheLock::~GPUKernelCompilerWindowProcessCacheLock() {}

bool GPUKernelCompilerWindowProcessCacheLock::acquired() const
{
	return false;
}

#endif // _WIN32
