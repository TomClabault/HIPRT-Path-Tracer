/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Compiler/HIPKernelCompiler.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

#include <chrono>
#include <deque>

oroFunction HIPKernelCompiler::compile_kernel(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, hiprtContext& hiprt_ctx, bool use_cache, const std::string& additional_cache_key)
{
	std::string kernel_file_path = kernel.get_kernel_file_path();
	std::string kernel_function_name = kernel.get_kernel_function_name();
	const std::vector<std::string>& additional_include_dirs = kernel_compiler_options->get_additional_include_directories();
	const std::vector<std::string>& compiler_options = kernel_compiler_options->get_relevant_macros_as_std_vector_string(kernel);

	std::cout << "Compiling kernel \"" << kernel_function_name << "\"..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	hiprtApiFunction trace_function_out;
	if (HIPPTOrochiUtils::build_trace_kernel(hiprt_ctx, kernel_file_path, kernel_function_name, trace_function_out, additional_include_dirs, compiler_options, 0, 1, use_cache, nullptr, additional_cache_key) != hiprtError::hiprtSuccess)
	{
		std::cerr << "Unable to compile kernel \"" << kernel_function_name << "\". Cannot continue." << std::endl;
		int ignored = std::getchar();
		std::exit(1);
	}

	std::cout << std::endl;

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "Kernel \"" << kernel_function_name << "\" compiled in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

	return *reinterpret_cast<oroFunction*>(&trace_function_out);
}

std::string HIPKernelCompiler::find_in_include_directories(const std::string& include_name, const std::vector<std::string>& include_directories)
{
	for (const std::string& include_directory : include_directories)
	{
		std::string add_slash = include_directory[include_directory.length() - 1] != '/' ? "/" : "";
		std::string file_path = include_directory + add_slash +  include_name;
		std::ifstream try_open_file(file_path);
		if (try_open_file.is_open())
			return file_path;
	}

	return "";
}

void HIPKernelCompiler::process_include(const std::string& include_file_path, const std::vector<std::string>& include_directories, std::unordered_set<std::string>& output_includes)
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
	{
		std::cerr << "Could not generate additional cache key for kernel with path: " << include_file_path << ". Error is: " << strerror(errno);
	}
}

std::string HIPKernelCompiler::get_additional_cache_key(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options)
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
		HIPKernelCompiler::process_include(current_file, kernel_compiler_options->get_additional_include_directories(), new_includes);

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
