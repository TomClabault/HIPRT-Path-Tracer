/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class GPUKernel
{
public:
	static const std::vector<std::string> COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS;

	GPUKernel();
	GPUKernel(const std::string& kernel_file_path, const std::string& kernel_function_name);

	std::string get_kernel_file_path() const;
	std::string get_kernel_function_name() const;

	void set_kernel_file_path(const std::string& kernel_file_path);
	void set_kernel_function_name(const std::string& kernel_function_name);

	void compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, std::vector<hiprtFuncNameSet> func_name_sets = {}, bool use_cache = true);
	void compile_silent(std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, std::vector<hiprtFuncNameSet> func_name_sets = {}, bool use_cache = true);
	void launch(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream);
	void launch_synchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, float* execution_time_out);
	void launch_asynchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream);

	/**
	 * Sets an additional macro that will be passed to the GPU compiler when compiling this kernel
	 */
	void add_additional_macro_for_compilation(const std::string& name, int value);

	/**
	 * Returns a vector of strings of the form { -DMacroName=value, ... } from all the additional
	 * macros that were added to this kernel by calling 'add_additional_macro_for_compilation()'
	 */
	std::vector<std::string> get_additional_compiler_macros() const;

	/**
	 * Reads the kernel file and all of its includes to find what option macros this kernel uses.
	 * 
	 * Calling this function update the m_used_option_macros member attribute.
	 */
	void parse_option_macros_used();

	/**
	 * Given an option macro name ("InteriorStackStrategy", "DirectLightSamplingStrategy", "EnvmapSamplingStrategy", ...
	 * for examples. They are all defined in KernelOptions.h), returns true if the kernel uses that option macro.
	 * False otherwise.
	 * 
	 * The kernel "uses" that macro if changing the value of that macro and recompiling the kernel
	 * changes the output of the compiler. For example, the camera ray kernel doesn't care about
	 * which direct lighting sampling strategy we're using. It also doesn't care about our envmap
	 * sampling strategy. So we way that the camera ray kernel doesn't use the
	 * "DirectLightSamplingStrategy" and "EnvmapSamplingStrategy" options macro
	 */
	bool uses_macro(const std::string& macro_name) const;

	/**
	 * Returns the number of GPU register that this kernel is using. This function
	 * must be called after the kernel has been compiled. 
	 * This function may also return 0 if the device doesn't support querrying
	 * the number of registers
	 */
	int get_kernel_attribute(oroFunction_attribute attribute);
	static int get_kernel_attribute(oroFunction compiled_kernel, oroFunction_attribute attribute);

	/**
	 * Returns the compiler options of this kernel so that they can be modified
	 */
	GPUKernelCompilerOptions& get_kernel_options();
	const GPUKernelCompilerOptions& get_kernel_options() const;

	/**
	 * Synchronizes the value of the options of this kernel with the values of the macros of 'other_options'.
	 * This means that if the value of the macro "MY_MACRO" is modified in 'other_options', the value of 'MY_MACRO'
	 * will also be modified in this kernel options.
	 * 
	 * Macros that are in the 'options_excluded" set will not be synchronized.
	 * 
	 * Macros that are present in 'other_options' but that are not present in this kernel's option
	 * will be added to this kernel and their vlaue will be synchronized with 'other_options'
	 * 
	 * This function can be useful if you want to have a global set of macros shared by multiple kernels. 
	 * You can thus synchronize all your kernel with that global set of macros and when it is modified, 
	 * all the kernels will use the new values.
	 */
	void synchronize_options_with(const GPUKernelCompilerOptions& other_options, const std::unordered_set<std::string>& options_excluded = {});

	/**
	 * Returns the time taken for the last execution of this kernel in milliseconds
	 */
	float get_last_execution_time();

	/**
	 * Structure used to pass data to the compute_elapsed_time_callback that computes the
	 * elapsed time between the start and end events of this structure and stores the elapsed
	 * time in 'elapsed_time_out'
	 */
	struct ComputeElapsedTimeCallbackData
	{
		// Start and end events to compute the elapsed time between
		oroEvent_t start, end;

		// The elapsed time will be stored in here in milliseconds
		float* elapsed_time_out;

		// Needed to set the CUDA/HIP context as current to be able to call
		// CUDA/HIP functions from the callback
		HIPRTOrochiCtx* hiprt_orochi_ctx;
	};

	bool has_been_compiled() const;

	bool is_precompiled() const;
	void set_precompiled(bool precompiled);

private:
	std::string m_kernel_file_path = "";
	std::string m_kernel_function_name = "";

	// Whether or not the events have been created yet.
	// We only create them on the first launch() call
	bool m_events_created = false;
	// GPU events to time the execution time
	oroEvent_t m_execution_start_event = nullptr;
	oroEvent_t m_execution_stop_event = nullptr;
	float m_last_execution_time = -1.0f;

	// Whether or not the macros used by this kernel have been modified recently.
	// Only adding new macros / removing macros invalidate the macros.
	// Changing the values of macros doesn't invalidate the macros.
	// This variable is used to determine whether or not we need to parse the kernel
	// source file to collect the macro actually used during the compilation of the kernel
	bool m_option_macro_invalidated = true;

	// Which option macros (as defined in KernelOptions.h) the kernel uses.
	// 
	// See uses_macro() for some examples of what "use" means.
	std::unordered_set<std::string> m_used_option_macros;

	// An additional map of macros to pass to the compiler for this kernel and their values.
	//
	// Example: { "ReSTIR_DI_InitialCandidatesKernel", 1 }
	std::unordered_map<std::string, int> m_additional_compilation_macros;

	// Options/macros used by the compiler when compiling this kernel
	GPUKernelCompilerOptions m_compiler_options;

	oroFunction m_kernel_function = nullptr;

	// If true, this means that this kernel is only used for precompilation and will be
	// discarded after it's been compiled
	// This is used in the GPUKernelCompiler to determine whether or not we should increment
	// the counter of the ImGuiLoggerLine that counts how many kernels have been precompiled
	// so far
	bool m_is_precompiled_kernel = false;
};

#endif
