/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <string>
#include <unordered_set>
#include <vector>

class GPUKernelCompilerOptions;

class GPUKernel
{
public:
	GPUKernel();
	GPUKernel(const std::string& kernel_file_path, const std::string& kernel_function_name);

	std::string get_kernel_file_path();
	std::string get_kernel_function_name();

	void set_kernel_file_path(const std::string& kernel_file_path);
	void set_kernel_function_name(const std::string& kernel_function_name);

	void compile(hiprtContext& hiprt_ctx, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, bool use_cache = true);
	void launch_timed_synchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, float* execution_time_out);
	void launch_timed_asynchronous(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, oroStream_t stream);

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
	 * Indicates that this kernel doesn't use the macro with the name given in parameter
	 */
	void unuse_macro(const std::string& macro_name);

	/**
	 * Returns the number of GPU register that this kernel is using. This function
	 * must be called after the kernel has been compiled. 
	 * This function may also return 0 if the device doesn't support querrying
	 * the number of registers
	 */
	int get_kernel_attribute(oroDeviceProp device_properties, oroFunction_attribute attribute);

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

	// Which option macros (as defined in KernelOptions.h) the kernel doesn't use. 
	// 
	// See uses_macro() for some examples of what "use" means.
	// 
	// A kernel uses all options macros by default. Call 'unuse_macro()'
	// if you wish the kernel not to use a given macro
	std::unordered_set<std::string> m_unused_option_macros;

	// Whether or not the events have been created yet.
	// We only create them on the first launch() call
	bool m_events_created = false;
	// GPU events to time the execution time
	oroEvent_t m_execution_start_event = nullptr;
	oroEvent_t m_execution_stop_event = nullptr;
	float m_last_execution_time = -1.0f;

	oroFunction m_kernel_function = nullptr;
};

#endif