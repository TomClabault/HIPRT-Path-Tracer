/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerWindowProcess.h"
#include "Compiler/GPUKernelCompilerWindowProcessCacheLock.h"
#include "Compiler/GPUKernelCompilerWindowProcessSerialization.h"
#include "Compiler/GPUKernel.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <system_error>

#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max

GPUKernelCompilerWindowProcessCompilationRequest GPUKernelCompilerWindowProcess::make_window_process_compilation_request(
	const GPUKernel& kernel,
	const std::vector<std::string>& additional_include_directories,
	const std::vector<std::string>& compiler_options,
	int num_geom_types,
	int num_ray_types,
	bool use_compiler_cache,
	hiprtFuncNameSet* function_name_sets,
	const std::string& additional_cache_key,
	int device_index)
{
	GPUKernelCompilerWindowProcessCompilationRequest request;
	request.kernel_file_path			   = kernel.get_kernel_file_path();
	request.kernel_function_name		   = kernel.get_kernel_function_name();
	request.additional_include_directories = additional_include_directories;
	request.compiler_options			   = compiler_options;
	request.num_geom_types				   = num_geom_types;
	request.num_ray_types				   = num_ray_types;
	request.use_compiler_cache			   = use_compiler_cache;
	request.has_function_name_sets		   = function_name_sets != nullptr;
	request.additional_cache_key		   = additional_cache_key;
	request.device_index				   = device_index;

	if (function_name_sets != nullptr && num_geom_types > 0 && num_ray_types > 0)
	{
		size_t function_name_set_count = static_cast<size_t>(num_geom_types) * static_cast<size_t>(num_ray_types);
		request.function_name_sets.reserve(function_name_set_count);

		for (size_t function_name_set_index = 0; function_name_set_index < function_name_set_count; function_name_set_index++)
		{
			GPUKernelCompilerWindowProcessFunctionNameSet request_function_name_set;
			const hiprtFuncNameSet& function_name_set = function_name_sets[function_name_set_index];

			if (function_name_set.intersectFuncName != nullptr)
				request_function_name_set.intersect_function_name = std::string(function_name_set.intersectFuncName);
			if (function_name_set.filterFuncName != nullptr)
				request_function_name_set.filter_function_name = std::string(function_name_set.filterFuncName);

			request.function_name_sets.push_back(request_function_name_set);
		}
	}

	return request;
}

std::wstring GPUKernelCompilerWindowProcess::get_current_executable_path()
{
	std::vector<wchar_t> executable_path_buffer(MAX_PATH);
	DWORD executable_path_length = 0;

	do
	{
		executable_path_length = GetModuleFileNameW(nullptr, executable_path_buffer.data(), static_cast<DWORD>(executable_path_buffer.size()));
		if (executable_path_length == 0)
			return L"";

		if (executable_path_length < executable_path_buffer.size() - 1)
			return std::wstring(executable_path_buffer.data(), executable_path_length);

		executable_path_buffer.resize(executable_path_buffer.size() * 2);
	} while (true);
}

std::wstring GPUKernelCompilerWindowProcess::quote_windows_argument(const std::wstring& argument)
{
	return L"\"" + argument + L"\"";
}

std::filesystem::path GPUKernelCompilerWindowProcess::make_request_file_path()
{
	static std::atomic<unsigned int> request_counter = 0;

	unsigned int request_number	   = request_counter.fetch_add(1);
	std::wstring request_file_name = L"HIPRTKernelCompile-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetCurrentThreadId()) + L"-" +
									 std::to_wstring(request_number) + L".bin";
	return std::filesystem::temp_directory_path() / request_file_name;
}

std::string GPUKernelCompilerWindowProcess::read_worker_output(void* output_read_handle)
{
	HANDLE read_handle = reinterpret_cast<HANDLE>(output_read_handle);
	std::string output;
	char buffer[4096];

	while (true)
	{
		DWORD bytes_read	= 0;
		BOOL read_succeeded = ReadFile(read_handle, buffer, sizeof(buffer), &bytes_read, nullptr);
		if (read_succeeded == FALSE || bytes_read == 0)
			break;

		output.append(buffer, bytes_read);
	}

	CloseHandle(read_handle);
	return output;
}

void GPUKernelCompilerWindowProcess::print_worker_output(const std::string& output)
{
	if (output.empty())
		return;

	static std::mutex output_mutex;
	std::lock_guard<std::mutex> lock(output_mutex);
	std::cout.write(output.data(), static_cast<std::streamsize>(output.size()));
	if (output.back() != '\n')
		std::cout << std::endl;
	else
		std::cout.flush();
}
#endif // _WIN32

bool GPUKernelCompilerWindowProcess::compile(const GPUKernelCompilerWindowProcessCompilationRequest& request)
{
#ifdef _WIN32
	try
	{
		std::filesystem::path request_file_path = GPUKernelCompilerWindowProcess::make_request_file_path();
		if (!GPUKernelCompilerWindowProcessSerialization::write_request(request_file_path, request))
		{
			std::error_code remove_error;
			std::filesystem::remove(request_file_path, remove_error);
			return false;
		}

		std::wstring current_executable = GPUKernelCompilerWindowProcess::get_current_executable_path();
		if (current_executable.empty())
		{
			std::error_code remove_error;
			std::filesystem::remove(request_file_path, remove_error);
			return false;
		}

		std::filesystem::path worker_executable_path = std::filesystem::path(current_executable).parent_path() / "GPUKernelCompilerWorker.exe";
		std::wstring command_line = GPUKernelCompilerWindowProcess::quote_windows_argument(worker_executable_path.wstring()) + L" --kernel-compiler-worker " +
									GPUKernelCompilerWindowProcess::quote_windows_argument(request_file_path.wstring());
		std::wstring current_directory = std::filesystem::current_path().wstring();
		GPUKernelCompilerWindowProcessCacheLock cache_lock(request);
		if (!cache_lock.acquired())
		{
			std::error_code remove_error;
			std::filesystem::remove(request_file_path, remove_error);
			return false;
		}

		STARTUPINFOW startup_info		 = {};
		startup_info.cb					 = sizeof(startup_info);
		startup_info.dwFlags			 = STARTF_USESTDHANDLES;
		PROCESS_INFORMATION process_info = {};

		SECURITY_ATTRIBUTES pipe_security_attributes = {};
		pipe_security_attributes.nLength			 = sizeof(pipe_security_attributes);
		pipe_security_attributes.bInheritHandle		 = TRUE;

		HANDLE output_read_handle  = nullptr;
		HANDLE output_write_handle = nullptr;
		if (CreatePipe(&output_read_handle, &output_write_handle, &pipe_security_attributes, 0) == FALSE)
		{
			std::error_code remove_error;
			std::filesystem::remove(request_file_path, remove_error);
			return false;
		}

		if (SetHandleInformation(output_read_handle, HANDLE_FLAG_INHERIT, 0) == FALSE)
		{
			CloseHandle(output_read_handle);
			CloseHandle(output_write_handle);
			std::error_code remove_error;
			std::filesystem::remove(request_file_path, remove_error);
			return false;
		}

		startup_info.hStdOutput = output_write_handle;
		startup_info.hStdError	= output_write_handle;
		BOOL process_created = CreateProcessW(worker_executable_path.wstring().c_str(), command_line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
											  current_directory.c_str(), &startup_info, &process_info);
		CloseHandle(output_write_handle);

		if (process_created == FALSE)
		{
			CloseHandle(output_read_handle);
			std::error_code remove_error;
			std::filesystem::remove(request_file_path, remove_error);
			return false;
		}

		std::string worker_output = GPUKernelCompilerWindowProcess::read_worker_output(output_read_handle);
		DWORD wait_result		  = WaitForSingleObject(process_info.hProcess, INFINITE);
		DWORD process_exit_code	  = 1;
		BOOL exit_code_read		  = GetExitCodeProcess(process_info.hProcess, &process_exit_code);
		CloseHandle(process_info.hThread);
		CloseHandle(process_info.hProcess);

		std::error_code remove_error;
		std::filesystem::remove(request_file_path, remove_error);
		bool worker_succeeded = wait_result == WAIT_OBJECT_0 && exit_code_read != FALSE && process_exit_code == 0;
		if (!worker_succeeded)
			GPUKernelCompilerWindowProcess::print_worker_output(worker_output);

		return worker_succeeded;
	}
	catch (const std::exception&)
	{
		return false;
	}
#else
	(void)request;
	return false;
#endif // _WIN32
}
