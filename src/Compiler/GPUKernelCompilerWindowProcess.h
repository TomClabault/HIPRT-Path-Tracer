/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_H
#define GPU_KERNEL_COMPILER_WINDOW_PROCESS_H

#include "Compiler/GPUKernelCompilerWindowProcessCompilationRequest.h"

#include <hiprt/hiprt.h>

#include <filesystem>
#include <string>

class GPUKernel;

class GPUKernelCompilerWindowProcess
{
public:
	/**
	 * Starts a worker process that compiles the request into the HIPRT shader cache.
	 *
	 * The worker does not return HIPRT function or module handles. Those handles are
	 * process-local, so the caller must load the resulting cache entry in its own
	 * HIPRT context after this function returns successfully.
	 */
	static bool compile(const GPUKernelCompilerWindowProcessCompilationRequest& request);

#ifdef _WIN32
	static GPUKernelCompilerWindowProcessCompilationRequest make_window_process_compilation_request(
		const GPUKernel& kernel,
		const std::vector<std::string>& additional_include_directories,
		const std::vector<std::string>& compiler_options,
		int num_geom_types,
		int num_ray_types,
		bool use_compiler_cache,
		hiprtFuncNameSet* function_name_sets,
		const std::string& additional_cache_key,
		int device_index);

	static std::string read_worker_output(void* output_read_handle);
	static void print_worker_output(const std::string& output);
#endif // _WIN32

private:
	static std::wstring get_current_executable_path();
	static std::wstring quote_windows_argument(const std::wstring& argument);
	static std::filesystem::path make_request_file_path();
};

#endif
