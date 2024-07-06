/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIP_KERNEL_H
#define HIP_KERNEL_H

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <string>
#include <vector>

class HIPKernel
{
public:
	HIPKernel() {};
	HIPKernel(const std::string& kernel_file_path, const std::string& kernel_function_name);

	std::string get_kernel_file_path();
	std::string get_kernel_function_name();
	std::vector<std::string> get_additional_include_directories();
	std::vector<std::string> get_compiler_options();

	void set_kernel_file_path(const std::string& kernel_file_path);
	void set_kernel_function_name(const std::string& kernel_function_name);
	void set_additional_includes(const std::vector<std::string>& additional_include_directories);
	void set_compiler_options(const std::vector<std::string>& compiler_options);

	void compile(hiprtContext& hiprt_ctx);
	void launch(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args);
	void launch_timed(int tile_size_x, int tile_size_y, int res_x, int res_y, void** launch_args, float* execution_time_out);

	/**
	 * Returns the number of GPU register that this kernel is using. This function
	 * must be called after the kernel has been compiled. 
	 * This function may also return 0 if the device doesn't support querrying
	 * the number of registers
	 */
	int get_register_count(oroDeviceProp device_properties);

private:
	std::string m_kernel_file_path = "";
	std::string m_kernel_function_name = "";

	std::vector<std::string> m_additional_include_directories;
	std::vector<std::string> m_compiler_options;

	// Whether or not the events have been created yet.
	// We only create them on the first launch() call
	bool m_events_created = false;
	// GPU events to time the execution time
	oroEvent_t m_execution_start_event = nullptr;
	oroEvent_t m_execution_stop_event = nullptr;

	oroFunction m_kernel_function = nullptr;
};

#endif
