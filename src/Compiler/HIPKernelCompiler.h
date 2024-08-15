/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIP_KERNEL_COMPILER_H
#define HIP_KERNEL_COMPILER_H

#include "Compiler/GPUKernel.h"

#include <unordered_set>

class HIPKernelCompiler
{
public:
	static oroFunction compile_kernel(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, hiprtContext& hiprt_ctx, bool use_cache, const std::string& additional_cache_key);

	/**
	 * Takes an include name ("Device/includes/MyInclude.h" for example) and a list of include directories.
	 * If the given include can be found in one the given include directories, the concatenation of the
	 * include directory with the include name is returned
	 * 
	 * If it cannot be found, the empty string is returned
	 */
	static std::string find_in_include_directories(const std::string& include_name, const std::vector<std::string>& include_directories);

	/**
	 * Reads the given file (include_file_path) and fills the 'output_includes' parameters with the includes used by that file. 
	 * Only includes that can be found in the given 'include_directories' will be added to the output parameter, others will be
	 * ignored.
	 */
	static void process_include(const std::string& include_file_path, const std::vector<std::string>& include_directories, std::unordered_set<std::string>& output_includes);

	/**
	 * Returns a string that consists of the concatenation of the include dependencies of the given kernel.
	 * For example, if the given kernel has includes "Include1.h" and "Include2.h" and that Include2.h itself
	 contains "Include3.h", the returned string will be the concatenation of the last modification time
	 (on the hard drive) of these 3 files, which may look something like this:
	 * 
	 * "133378594621" + "13334848655" + "1331849841" = "133378594621133348486551331849841".
	 * 
	 * Note that the timestamps are "time since epoch" so that's why they're pretty unintelligible.
	 * 
	 * The returned string, so-called "additional cache key", can be used to determine whether or not a GPU
	 * shader needs to be recompiled or not by passing it to the HIPRT compiler which will take it into account
	 * into the hash used to determine whether a file is up to date or not.
	 * 
	 * Note that only includes that can be found in the additional include directories of the given kernel
	 * (the parameter of this function) are going to be considered for the concatenation of time stamps. Includes
	 * that cannot be found in the kernel's include directories are ignored (this prevents the issue of losing
	 * ourselves in the parsing of stdlib headers for example. stdlib headers will be ignored since they are not
	 * [probably] in the include directories of the kernel).
	 */
	static std::string get_additional_cache_key(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options);
};

#endif
