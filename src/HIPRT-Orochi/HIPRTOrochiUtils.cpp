/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Utils/Utils.h"

#include <deque>
#include <unordered_set>

void orochi_check_error(oroError res, const char* file, uint32_t line)
{
	if (res != oroSuccess)
	{
		const char* msg;
		oroGetErrorString(res, &msg);
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		Utils::debugbreak();
		exit(EXIT_FAILURE);
	}
}

void orochi_rtc_check_error(orortcResult res, const char* file, uint32_t line)
{
	if (res != ORORTC_SUCCESS)
	{
		std::cerr << "ORORTC error: '" << orortcGetErrorString(res) << "' [ " << res << " ] on line " << line << " "
			<< " in '" << file << "'." << std::endl;

		Utils::debugbreak();
		exit(EXIT_FAILURE);
	}
}

void hiprt_check_error(hiprtError res, const char* file, uint32_t line)
{
	if (res != hiprtSuccess)
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		Utils::debugbreak();
		exit(EXIT_FAILURE);
	}
}

namespace HIPPTOrochiUtils
{
	std::string locate_header_in_include_dirs(const std::string& header_name, const std::vector<std::string> include_directories)
	{
		for (const std::string& additional_include : include_directories)
		{
			std::string path = additional_include + header_name;
			std::fstream f(path);
			if (f.is_open())
				return path;
		}

		return std::string();
	}

	bool read_source_code(const std::string& path, std::string& sourceCode, std::vector<std::string>* includes)
	{
		std::fstream f(path);
		if (f.is_open())
		{
			size_t sizeFile;
			f.seekg(0, std::fstream::end);
			size_t size = sizeFile = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);
			if (includes)
			{
				sourceCode.clear();
				std::string line;
				char buf[512];
				while (std::getline(f, line))
				{
					if (strstr(line.c_str(), "#include") != 0)
					{
						const char* a = strstr(line.c_str(), "<");
						const char* b = strstr(line.c_str(), ">");
						if (!a)
						{
							// If we couldn't find a "<", trying to find a '"'
							a = strstr(line.c_str(), "\"");
							if (!a)
							{
								// Not even '"' was find, that's invalid #include syntax
								std::cerr << "Unable to parse header name in line: " << line << std::endl;

								continue;
							}
						}

						// Same thing with the ending character, '>' or another '"'
						if (!b)
						{
							b = strstr(a + 1, "\"");

							if (!b)
							{
								std::cerr << "Unable to parse header name in line: " << line << std::endl;

								continue;
							}
						}

						int n = b - a - 1;
						memcpy(buf, a + 1, n);
						buf[n] = '\0';
						includes->push_back(buf);
					}

					sourceCode += line + '\n';
				}
			}
			else
			{
				sourceCode.resize(size, ' ');
				f.read(&sourceCode[0], size);
			}
			f.close();
			return true;
		}
		return false;
	}

	hiprtError build_trace_kernel(hiprtContext ctxt,
		const std::string& kernel_file_path,
		const char* function_name,
		hiprtApiFunction& kernel_function_out,
		const std::vector<std::string>& additional_include_directories,
		const std::vector<std::string>& compiler_options,
		unsigned int num_geom_types, unsigned int num_ray_types, 
		bool use_compiler_cache,
		hiprtFuncNameSet* func_name_set)
	{
		std::vector<std::string> include_names;
		std::string kernel_source_code;
		read_source_code(kernel_file_path, kernel_source_code, &include_names);

		std::vector<const char*> compiler_options_cstr;
		
		for (const std::string& option : compiler_options)
			compiler_options_cstr.push_back(option.c_str());

		// Adding the additional include directories as options for the GPU compiler (-I flag before the include directory path)
		std::vector<std::string> additional_includes_str;
		additional_includes_str.reserve(additional_include_directories.size());
		for (const std::string& additional_include_dir : additional_include_directories)
		{
			additional_includes_str.push_back("-I" + additional_include_dir);
			compiler_options_cstr.push_back(additional_includes_str.back().c_str());
		}

		return hiprtBuildTraceKernels(
			ctxt,
			1,
			&function_name,
			kernel_source_code.c_str(),
			kernel_file_path.c_str(),
			0,
			nullptr,
			nullptr,
			compiler_options_cstr.size(),
			compiler_options_cstr.size() > 0 ? compiler_options_cstr.data() : nullptr,
			num_geom_types,
			num_ray_types,
			func_name_set,
			&kernel_function_out,
			nullptr,
			use_compiler_cache);
	}
}
