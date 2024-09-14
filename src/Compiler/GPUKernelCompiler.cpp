/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Compiler/GPUKernelCompiler.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

#include <chrono>
#include <deque>
#include <mutex>

GPUKernelCompiler g_gpu_kernel_compiler;

oroFunction_t GPUKernelCompiler::compile_kernel(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, HIPRTOrochiCtx& hiprt_orochi_ctx, bool use_cache, const std::string& additional_cache_key)
{
	std::string kernel_file_path = kernel.get_kernel_file_path();
	std::string kernel_function_name = kernel.get_kernel_function_name();
	const std::vector<std::string>& additional_include_dirs = kernel_compiler_options->get_additional_include_directories();
	std::vector<std::string> compiler_options = kernel_compiler_options->get_relevant_macros_as_std_vector_string(kernel);

	auto start = std::chrono::high_resolution_clock::now();

	hiprtApiFunction trace_function_out;
	if (HIPPTOrochiUtils::build_trace_kernel(hiprt_orochi_ctx.hiprt_ctx, kernel_file_path, kernel_function_name, trace_function_out, additional_include_dirs, compiler_options, 0, 1, use_cache, nullptr, additional_cache_key) != hiprtError::hiprtSuccess)
	{
		std::cerr << "Unable to compile kernel \"" << kernel_function_name << "\". Cannot continue." << std::endl;
		int ignored = std::getchar();
		std::exit(1);
	}

	oroFunction kernel_function = reinterpret_cast<oroFunction>(trace_function_out);

	auto stop = std::chrono::high_resolution_clock::now();
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		std::cout << "Kernel \"" << kernel_function_name << "\" compiled in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms. ";

		if (hiprt_orochi_ctx.device_properties.major >= 7 && hiprt_orochi_ctx.device_properties.minor >= 5)
			// Getting the registers of a kernel only seems to be available on 7.5 and above 
			// (works on a 2060S but doesn't on a GTX 970 or GTX 1060, couldn't try more hardware 
			// so maybe 7.5 is too conservative)
			std::cout << GPUKernel::get_kernel_attribute(kernel_function, ORO_FUNC_ATTRIBUTE_NUM_REGS) << " registers." << std::endl;
		else
			std::cout << std::endl;
	}

	return kernel_function;
}

std::string GPUKernelCompiler::find_in_include_directories(const std::string& include_name, const std::vector<std::string>& include_directories)
{
	for (const std::string& include_directory : include_directories)
	{
		std::string add_slash = include_directory[include_directory.length() - 1] != '/' ? "/" : "";
		std::string file_path = include_directory + add_slash + include_name;
		std::ifstream try_open_file(file_path);
		if (try_open_file.is_open())
			return file_path;
	}

	return "";
}

void GPUKernelCompiler::read_includes_of_file(const std::string& include_file_path, const std::vector<std::string>& include_directories, std::unordered_set<std::string>& output_includes)
{
	std::ifstream include_file(include_file_path);
	if (include_file.is_open())
	{
		std::string line;
		while (std::getline(include_file, line))
		{
			if (line.starts_with("#include "))
			{
				size_t find_start = line.find('<');
				if (find_start == std::string::npos)
				{
					// Trying to find a quote instead
					find_start = line.find('"');
					if (find_start == std::string::npos)
						// Couldn't find a quote either, ill-formed include
						continue;
				}

				size_t find_end = line.rfind('>');
				if (find_end == std::string::npos)
				{
					// Trying to find a quote instead
					find_end = line.rfind('"');
					if (find_end == std::string::npos)
						// Couldn't find a quote either, ill-formed include
						continue;
				}

				// We found the include string, now we're going to check whether it can be found
				// in the given includes directories (which contain the only includes that we're
				// interested in)

				// Include file with leading Device/includes/... or whatever folder the include may come from
				std::string full_include_name = line.substr(find_start + 1, find_end - find_start - 1);

				// We have only the file name (which looks like "MyInclude.h" for example), let's see
				// if it can be found in the include directories
				std::string include_file_path = find_in_include_directories(full_include_name, include_directories);

				if (!include_file_path.empty())
					// Adding to file path that can directly be opened in an std::ifstream
					output_includes.insert(include_file_path);
			}
			else
				continue;
		}

	}
	else
		std::cerr << "Could not generate additional cache key for kernel with path: " << include_file_path << ". Error is: " << strerror(errno);
}

std::unordered_set<std::string> GPUKernelCompiler::read_option_macro_of_file(const std::string& filepath)
{
	std::string file_modification_time;

	try
	{
		std::chrono::time_point modification_time = std::filesystem::last_write_time(filepath);

		file_modification_time = std::to_string(modification_time.time_since_epoch().count());
	}
	catch (std::filesystem::filesystem_error e)
	{
		std::cerr << "HIPKernelCompiler - Unable to open include file \"" << filepath << "\" for option macros analyzing: " << e.what() << std::endl;

		return std::unordered_set<std::string>();
	}

	auto cache_timestamp_find = m_filepath_to_options_macros_cache_timestamp.find(filepath);
	if (cache_timestamp_find != m_filepath_to_options_macros_cache_timestamp.end() && cache_timestamp_find->second == file_modification_time)
	{
		// Cache hit
		return m_filepath_to_option_macros_cache[filepath];
	}

	std::unordered_set<std::string> option_macros;
	std::ifstream include_file(filepath);
	if (include_file.is_open())
	{
		std::string line;
		while (std::getline(include_file, line))
			for (const std::string& existing_macro_option : GPUKernelCompilerOptions::ALL_MACROS_NAMES)
				if (line.find(existing_macro_option) != std::string::npos)
					option_macros.insert(existing_macro_option);

	}
	else
		std::cerr << "Could not open file " << filepath << " for reading option macros used by that file. Error is : " << strerror(errno);

	// The cache is shared to all threads using this GPUKernelCompiler so we're locking that operation
	std::lock_guard<std::mutex> lock(m_mutex);

	// Updating the cache
	m_filepath_to_option_macros_cache[filepath] = option_macros;
	m_filepath_to_options_macros_cache_timestamp[filepath] = file_modification_time;

	return option_macros;
}

std::string GPUKernelCompiler::get_additional_cache_key(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options)
{
	std::unordered_set<std::string> already_processed_includes;
	std::deque<std::string> yet_to_process_includes;
	yet_to_process_includes.push_back(kernel.get_kernel_file_path());

	while (!yet_to_process_includes.empty())
	{
		std::string current_file = yet_to_process_includes.front();
		yet_to_process_includes.pop_front();

		if (already_processed_includes.find(current_file) != already_processed_includes.end())
			// We've already processed that file
			continue;

		already_processed_includes.insert(current_file);

		std::unordered_set<std::string> new_includes;
		read_includes_of_file(current_file, kernel_compiler_options->get_additional_include_directories(), new_includes);

		for (const std::string& new_include : new_includes)
			yet_to_process_includes.push_back(new_include);
	}

	// The cache key is going to be the concatenation of the last modified times of all the includes
	// that the kernel file we just parsed depends on. That way, if any dependency of this kernel has
	// been modified, the cache key will be different and the cache will be invalidated.
	std::string final_cache_key = "";
	for (const std::string& include : already_processed_includes)
	{
		// TODO this exception here should probably go up a level so that we can know that the kernel compilation failed --> set the kernel function to nullptr --> do try to launch the kernel (otherwise this will probably crash the driver)
		try
		{
			std::chrono::time_point modification_time = std::filesystem::last_write_time(include);

			final_cache_key += std::to_string(modification_time.time_since_epoch().count());
		}
		catch (std::filesystem::filesystem_error e)
		{
			std::cerr << "HIPKernelCompiler - Unable to open include file \"" << include << "\" for shader cache validation: " << e.what() << std::endl;

			return "";
		}
	}

	return final_cache_key;
}

std::unordered_set<std::string> GPUKernelCompiler::get_option_macros_used_by_kernel(const GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options)
{
	std::unordered_set<std::string> already_processed_includes;
	std::deque<std::string> yet_to_process_includes;
	yet_to_process_includes.push_back(kernel.get_kernel_file_path());

	while (!yet_to_process_includes.empty())
	{
		std::string current_file = yet_to_process_includes.front();
		yet_to_process_includes.pop_front();

		if (already_processed_includes.find(current_file) != already_processed_includes.end())
			// We've already processed that file
			continue;
		else if (current_file.find("KernelOptions.h") != std::string::npos)
			// Ignoring kernel options when looking for option macros
			continue;
		else if (current_file.find("Device/") == std::string::npos && current_file.find("HostDeviceCommon/") == std::string::npos)
			// Excluding files that are not in the Device/ or HostDeviveCommon/ folder because we're only
			// interested in kernel files, not CPU C++ files
			continue;

		already_processed_includes.insert(current_file);

		std::unordered_set<std::string> new_includes;
		read_includes_of_file(current_file, kernel_compiler_options->get_additional_include_directories(), new_includes);

		for (const std::string& new_include : new_includes)
			yet_to_process_includes.push_back(new_include);
	}

	std::unordered_set<std::string> option_macro_names;
	for (const std::string& include : already_processed_includes)
	{
		std::unordered_set<std::string> include_option_macros = read_option_macro_of_file(include);

		for (const std::string& option_macro : include_option_macros)
			option_macro_names.insert(option_macro);
	}

	return option_macro_names;
}
