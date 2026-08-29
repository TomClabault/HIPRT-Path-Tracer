/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_COMPILATION_REQUEST_H
#define GPU_KERNEL_COMPILER_WINDOW_PROCESS_COMPILATION_REQUEST_H

#include "Compiler/GPUKernelCompilerWindowProcessFunctionNameSet.h"

#include <string>
#include <vector>

struct GPUKernelCompilerWindowProcessCompilationRequest
{
	std::string kernel_file_path;
	std::string kernel_function_name;

	std::vector<std::string> additional_include_directories;
	std::vector<std::string> compiler_options;
	bool use_compiler_cache		= false;
	bool has_function_name_sets = false;

	int num_geom_types = 0;
	int num_ray_types  = 0;
	std::vector<GPUKernelCompilerWindowProcessFunctionNameSet> function_name_sets;
	std::string additional_cache_key;

	int device_index = 0;
};

#endif // #ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_COMPILATION_REQUEST_H
