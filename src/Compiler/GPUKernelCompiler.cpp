/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Compiler/GPUKernelCompiler.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

GPUKernelCompiler g_gpu_kernel_compiler;
extern ImGuiLogger g_imgui_logger;

// This variable will be initialized before the main function by the main thread
std::thread::id g_priority_thread_id = std::this_thread::get_id();
// Whether or not the main thread is currently compiling. Used in the condition variable.
// If the main thread is currently compiling (very likely that his was asked by the user through the UI), 
// other threads may not compile to give the user the priority for the compilation
bool g_main_thread_compiling = false;
bool g_background_shader_compilation_enabled = true;
std::condition_variable g_condition_for_compilation;

void enable_compilation_warnings(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, std::vector<std::string>& compiler_options)
{
	if (std::string(hiprt_orochi_ctx->device_properties.name).find("NVIDIA") == std::string::npos)
	{
		// AMD compiler options

		compiler_options.push_back("-Wall");
		compiler_options.push_back("-Weverything");
		compiler_options.push_back("-Wno-reorder-ctor");
		compiler_options.push_back("-Wno-c++98-compat");
		compiler_options.push_back("-Wno-c++98-compat-pedantic");
		compiler_options.push_back("-Wno-reserved-macro-identifier");
		compiler_options.push_back("-Wno-extra-semi-stmt");
		compiler_options.push_back("-Wno-reserved-identifier");
		compiler_options.push_back("-Wno-reserved-identifier");
		compiler_options.push_back("-Wno-float-conversion");
		compiler_options.push_back("-Wno-implicit-float-conversion");
		compiler_options.push_back("-Wno-implicit-int-float-conversion");
		compiler_options.push_back("-Wno-deprecated-copy-with-user-provided-copy");
		compiler_options.push_back("-Wno-disabled-macro-expansion");
		compiler_options.push_back("-Wno-float-equal");
		compiler_options.push_back("-Wno-sign-compare");
		compiler_options.push_back("-Wno-padded");
		compiler_options.push_back("-Wno-sign-conversion");
		compiler_options.push_back("-Wno-gnu-zero-variadic-macro-arguments");
		compiler_options.push_back("-Wno-missing-variable-declarations");
	}
}

oroFunction_t GPUKernelCompiler::compile_kernel(GPUKernel& kernel, const GPUKernelCompilerOptions& kernel_compiler_options, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, hiprtFuncNameSet* function_name_sets, int num_geom_types, int num_ray_types, bool use_cache, const std::string& additional_cache_key, bool silent)
{
	std::string kernel_file_path = kernel.get_kernel_file_path();
	std::string kernel_function_name = kernel.get_kernel_function_name();
	const std::vector<std::string>& additional_include_dirs = GPUKernel::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS;
	std::vector<std::string> compiler_options = kernel_compiler_options.get_relevant_macros_as_std_vector_string(&kernel);

	compiler_options.push_back("-g");
	compiler_options.push_back("-ggdb");
	
	enable_compilation_warnings(hiprt_orochi_ctx, compiler_options);

	// Locking because neither NVIDIA or AMD cannot compile kernels on multiple threads so we may as well
	// lock here to have better control on when to compile a kernel as well as have proper compilation times
	std::unique_lock<std::mutex> lock(m_compile_mutex);

	if (std::this_thread::get_id() != g_priority_thread_id)
		// Other threads wait if the main thread is compiling
		g_condition_for_compilation.wait(lock, []() { return !g_main_thread_compiling && g_background_shader_compilation_enabled; });

	auto start = std::chrono::high_resolution_clock::now();

	hiprtApiFunction trace_function_out;
	bool use_shader_cache;
	if (m_shader_cache_force_usage == GPUKernelCompiler::ShaderCacheUsageOverride::FORCE_SHADER_CACHE_OFF)
		use_shader_cache = false;
	else if (m_shader_cache_force_usage == GPUKernelCompiler::ShaderCacheUsageOverride::FORCE_SHADER_CACHE_ON)
		use_shader_cache = true;
	else
		use_shader_cache = use_cache;

	hiprtError compile_status = HIPPTOrochiUtils::build_trace_kernel(hiprt_orochi_ctx->hiprt_ctx, kernel_file_path, kernel_function_name, trace_function_out, additional_include_dirs, compiler_options, num_geom_types, num_ray_types, use_shader_cache, function_name_sets, additional_cache_key);
	if (compile_status != hiprtError::hiprtSuccess)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Unable to compile kernel \"%s\". Cannot continue.", kernel_function_name.c_str());

		return nullptr;
	}

	oroFunction kernel_function = reinterpret_cast<oroFunction>(trace_function_out);

	if (kernel.is_precompiled())
	{
		// Updating the logs
		m_precompiled_kernels_compilation_ended++;

		g_imgui_logger.update_line(ImGuiLogger::BACKGROUND_KERNEL_COMPILATION_LINE_NAME, "Compiling kernel permutations in the background... [%d / %d]", m_precompiled_kernels_compilation_ended.load(), m_precompiled_kernels_parsing_started.load());
	}

	auto stop = std::chrono::high_resolution_clock::now();

	if (!silent)
	{
		// Setting the current context is necessary because getting
		// functions attributes necessitates calling CUDA/HIP functions
		// which need their context to be current if not calling from
		// the main thread (which we are not if we are compiling kernels on multithreads)
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(hiprt_orochi_ctx->orochi_ctx));

		int nb_reg = GPUKernel::get_kernel_attribute(kernel_function, ORO_FUNC_ATTRIBUTE_NUM_REGS);
		int nb_shared = GPUKernel::get_kernel_attribute(kernel_function, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
		int nb_local = GPUKernel::get_kernel_attribute(kernel_function, ORO_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Kernel \"%s\" compiled in %ldms.\n\t[Reg, Shared, Local] = [%d, %d, %d]\n", kernel_function_name.c_str(), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(), nb_reg, nb_shared, nb_local);
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
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Could not generate additional cache key for kernel with path \"%s\": %s", include_file_path.c_str(), strerror(errno));

		Utils::debugbreak();
	}
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
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "HIPKernelCompiler - Unable to open include file \"%s\" for option macros analyzing: %s", filepath.c_str(), e.what());

		return std::unordered_set<std::string>();
	}

	{
		// We don't to read into the cache while someone may be writing to it (at the end of this function)
		// so we lock
		std::lock_guard<std::mutex> lock(m_option_macro_cache_mutex);

		auto cache_timestamp_find = m_filepath_to_options_macros_cache_timestamp.find(filepath);
		if (cache_timestamp_find != m_filepath_to_options_macros_cache_timestamp.end() && cache_timestamp_find->second == file_modification_time)
		{
			// Cache hit
			return m_filepath_to_option_macros_cache[filepath];
		}
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
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Could not open file \"%s\" for reading option macros used by that file: %s", filepath.c_str(), strerror(errno));

	// The cache is shared to all threads using this GPUKernelCompiler so we're locking that operation
	// The lock is destroyed when the function returns
	std::lock_guard<std::mutex> lock(m_option_macro_cache_mutex);

	// Updating the cache
	m_filepath_to_option_macros_cache[filepath] = option_macros;
	m_filepath_to_options_macros_cache_timestamp[filepath] = file_modification_time;

	return option_macros;
}

std::string GPUKernelCompiler::get_additional_cache_key(GPUKernel& kernel)
{
	m_additional_cache_key_started++;

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
		read_includes_of_file(current_file, GPUKernel::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS, new_includes);

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
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "HIPKernelCompiler - Unable to open include file \"%s\" for shader cache validation: %s", include.c_str(), e.what());

			m_additional_cache_key_ended++;
			// Notifying the condition variable that's used to
			// avoid exiting the application with ongoing IO operations
			m_read_macros_cv.notify_all();

			return "";
		}
	}

	m_additional_cache_key_ended++;
	// Notifying the condition variable that's used to
	// avoid exiting the application with ongoing IO operations
	m_read_macros_cv.notify_all();

	return final_cache_key;
}

std::unordered_set<std::string> GPUKernelCompiler::get_option_macros_used_by_kernel(const GPUKernel& kernel)
{
	if (kernel.is_precompiled())
		// If this kernel is being precompiled, we can increment the counter
		// used for logging
		m_precompiled_kernels_parsing_started++;

	// Limiting the number of threads that can get in here at the same time otherwise we may
	// get some "Too many files open!" error
	m_read_macros_semaphore.acquire();

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
		else if (current_file.find("HostDeviceCommon/KernelOptions") != std::string::npos)
			// Ignoring kernel options files when looking for option macros
			continue;
		else if (current_file.find("Device/") == std::string::npos && current_file.find("HostDeviceCommon/") == std::string::npos)
			// Excluding files that are not in the Device/ or HostDeviceCommon/ folder because we're only
			// interested in kernel files, not CPU C++ files
			continue;

		already_processed_includes.insert(current_file);

		std::unordered_set<std::string> new_includes;
		read_includes_of_file(current_file, GPUKernel::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS, new_includes);

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

	m_read_macros_semaphore.release();
	m_read_macros_cv.notify_all();

	if (kernel.is_precompiled())
	{
		// If this kernel is being precompiled, we can increment the counter
		// used for logging
		m_precompiled_kernels_parsing_ended++;

		// And update the log line
		g_imgui_logger.update_line(ImGuiLogger::BACKGROUND_KERNEL_PARSING_LINE_NAME, "Parsing kernel permutations in the background... [%d / %d]", m_precompiled_kernels_parsing_ended.load(), m_precompiled_kernels_parsing_started.load());
	}


	return option_macro_names;
}

void GPUKernelCompiler::wait_compiler_file_operations()
{
	std::mutex mutex;
	std::unique_lock<std::mutex> lock(mutex);

	m_read_macros_cv.wait(lock, [this]() { return m_precompiled_kernels_parsing_started == m_precompiled_kernels_parsing_ended; });
	m_read_macros_cv.wait(lock, [this]() { return m_additional_cache_key_started == m_additional_cache_key_ended; });
}

GPUKernelCompiler::ShaderCacheUsageOverride GPUKernelCompiler::get_shader_cache_usage_override() const
{
	return m_shader_cache_force_usage;
}

void GPUKernelCompiler::set_shader_cache_usage_override(GPUKernelCompiler::ShaderCacheUsageOverride override_usage)
{
	m_shader_cache_force_usage = override_usage;
}

