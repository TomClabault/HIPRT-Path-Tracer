/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_FUNCTION_NAME_SET_H
#define GPU_KERNEL_COMPILER_WINDOW_PROCESS_FUNCTION_NAME_SET_H

#include <optional>
#include <string>

struct GPUKernelCompilerWindowProcessFunctionNameSet
{
	std::optional<std::string> intersect_function_name;
	std::optional<std::string> filter_function_name;
};

#endif // #ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_FUNCTION_NAME_SET_H
