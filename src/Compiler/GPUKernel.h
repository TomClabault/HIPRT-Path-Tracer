/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class GPUKernelCompilerOptions;

class GPUKernel
{
public:
	GPUKernel();
	GPUKernel(const std::string& kernel_file_path, const std::string& kernel_function_name);

	std::string get_kernel_file_path() const;
	std::string get_kernel_function_name() const;

	void set_kernel_file_path(const std::string& kernel_file_path);
	void set_kernel_function_name(const std::string& kernel_function_name);

	void compile(hiprtContext& hiprt_ctx, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, bool use_cache = true);
	void launch_timed_synchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, float* execution_time_out);
	void launch_timed_asynchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream);

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
	void parse_option_macros_used(std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options);

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
	int get_kernel_attribute(oroDeviceProp device_properties, oroFunction_attribute attribute);

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
	};

	/**
	 * Callback function that can be given to oroLaunchHostFunc
	 * to compute the elapsed time between two events.
	 * 
	 * Data is expected to be a pointer to a *dynamically allocated* ComputeElapsedTimeCallbackData
	 * instance (i.e. with a 'new' call) whose fields have been properly set-up.
	 */
	static void compute_elapsed_time_callback(void* data);

private:
	void launch(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream);

	std::string m_kernel_file_path = "";
	std::string m_kernel_function_name = "";

	// Whether or not the events have been created yet.
	// We only create them on the first launch() call
	bool m_events_created = false;
	// GPU events to time the execution time
	oroEvent_t m_execution_start_event = nullptr;
	oroEvent_t m_execution_stop_event = nullptr;
	float m_last_execution_time = -1.0f;

	// Whether or not we have already parsed the kernel file to see
	// what option macro it uses or not
	bool m_option_macro_parsed = false;

	// Which option macros (as defined in KernelOptions.h) the kernel uses.
	// 
	// See uses_macro() for some examples of what "use" means.
	std::unordered_set<std::string> m_used_option_macros;

	// An additional map of macros to pass to the compiler for this kernel and their values.
	//
	// Example: { "ReSTIR_DI_InitialCandidatesKernel", 1 }
	std::unordered_map<std::string, int> m_additional_compilation_macros;

	oroFunction m_kernel_function = nullptr;
};

#endif
