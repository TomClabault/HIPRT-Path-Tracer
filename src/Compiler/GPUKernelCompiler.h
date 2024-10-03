/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_H
#define GPU_KERNEL_COMPILER_H

#include "Compiler/GPUKernel.h"

#include <mutex>
#include <semaphore>
#include <unordered_map>
#include <unordered_set>

class GPUKernelCompiler
{
public:
	oroFunction_t compile_kernel(GPUKernel& kernel, const GPUKernelCompilerOptions& kernel_compiler_options, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, hiprtFuncNameSet* function_name_sets, bool use_cache, const std::string& additional_cache_key, bool silent = false);

	/**
	 * Takes an include name ("Device/includes/MyInclude.h" for example) and a list of include directories.
	 * If the given include can be found in one the given include directories, the concatenation of the
	 * include directory with the include name is returned
	 *
	 * If it cannot be found, the empty string is returned
	 */
	std::string find_in_include_directories(const std::string& include_name, const std::vector<std::string>& include_directories);

	/**
	 * Reads the given file (include_file_path) and fills the 'output_includes' parameters with the includes (#include "XXX" pr #include <XXX>) 
	 * used by that file.
	 * 
	 * Only includes that can be found in the given 'include_directories' will be added to the output parameter, others will be
	 * ignored.
	 */
	void read_includes_of_file(const std::string& include_file_path, const std::vector<std::string>& include_directories, std::unordered_set<std::string>& output_includes);

	/**
	 * Returns a list of the option macro used in the given file.
	 * "Used" means that the option macro is used in #if, #ifdef or equivalent directives
	 */
	std::unordered_set<std::string> read_option_macro_of_file(const std::string& filepath);

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
	std::string get_additional_cache_key(GPUKernel& kernel);

	/**
	 * Returns a list of the option macro names used by the given kernel.
	 * 
	 * For example, this function will return {"DirectLightSamplingStrategy", "EnvmapSamplingStrategy"}
	 * if the given kernel uses this two macros (if the kernel has some "#if == DirectLightSamplingStrategy", "#ifdef DirectLightSamplingStrategy"
	 * directives or similar in its code)
	 */
	std::unordered_set<std::string> get_option_macros_used_by_kernel(const GPUKernel& kernel);

private:
	// Cache that maps a filepath to the option macros that it contains.
	// This saves us having to reparse the file to find the options macros
	// if the file was already parsed for another kernel by this GPUKernelCompiler
	std::unordered_map<std::string, std::unordered_set<std::string>> m_filepath_to_option_macros_cache;
	// Maps filepath to the last modification time of the file pointed by the filepath. 
	// Useful to invalidate the cache if the file was modified (meaning that the option
	// macros used by that file may have changed so we have to reparse the file)
	std::unordered_map<std::string, std::string> m_filepath_to_options_macros_cache_timestamp;

	// Because this GPUKernelCompiler may be used by multiple threads at the same time,
	// we may use that mutex sometimes to protect from race conditions
	std::mutex m_mutex;

	// Semaphore used by 'get_option_macros_used_by_kernel' so that not too many threads
	// read kernel files at the same time: this can cause a "Too many files open" error
	// 
	// Limiting to a maximum of 16 threads at a time
	std::counting_semaphore<> m_read_macros_semaphore { 16 };
};

#endif
