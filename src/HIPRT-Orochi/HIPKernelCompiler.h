/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIP_KERNEL_COMPILER_H
#define HIP_KERNEL_COMPILER_H

#include "HIPRT-Orochi/HIPKernel.h"

#include <unordered_set>

class HIPKernelCompiler
{
public:
	static oroFunction compile_kernel(HIPKernel& kernel, hiprtContext& hiprt_ctx, bool use_cache = true, const std::string& additional_cache_key = "");

	static std::string find_in_include_directories(const std::string& file_name, const std::vector<std::string>& include_directories);
	static void process_include(const std::string& include_file_path, const std::vector<std::string>& include_directories, std::unordered_set<std::string>& output_includes);
	static std::string get_additional_cache_key(HIPKernel& kernel);
};

#endif
