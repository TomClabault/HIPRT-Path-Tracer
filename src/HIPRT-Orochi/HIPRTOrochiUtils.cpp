/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

#include <deque>
#include <unordered_set>

void orochi_check_error(oroError res, const char* file, uint32_t line)
{
	if (res != oroSuccess)
	{
		const char* msg;
		oroGetErrorString(res, &msg);
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " " << " in '" << file << "'." << std::endl;

		exit(EXIT_FAILURE);
	}
}

void orochi_rtc_check_error(orortcResult res, const char* file, uint32_t line)
{
	if (res != ORORTC_SUCCESS)
	{
		std::cerr << "ORORTC error: '" << orortcGetErrorString(res) << "' [ " << res << " ] on line " << line << " "
			<< " in '" << file << "'." << std::endl;
		exit(EXIT_FAILURE);
	}
}

void hiprt_check_error(hiprtError res, const char* file, uint32_t line)
{
	if (res != hiprtSuccess)
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " " << " in '" << file << "'." << std::endl;

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
		const std::optional<std::vector<const char*>>& compiler_options,
		unsigned int num_geom_types, unsigned int num_ray_types, 
		bool use_compiler_cache)
	{
		std::vector<std::string> include_names;
		std::string kernel_source_code;
		read_source_code(kernel_file_path, kernel_source_code, &include_names);

		// Header parsing was causing issues with NVRTC and isn't even used anyway so disabling it for now...
		// 
		// 
		// 
		//// hiprtBuildTraceKernels() below needs a std::vector of const char*
		//// but OrochiUtils takes a vector of std::string as argument so we need both
		//// One could say that we could only store the std::string.c_str() into the vector
		//// of const char* but this is not valid since to string returned by std::string::c_str()
		//// is destroyed when the std::string is destroyed i.e. we need to keep the std::string
		//// alive to be able to use its c_str() const char* version so we do actually need
		//// both vectors (the std::string vectors being mainly used to keep the std::string alive)
		//std::vector<std::string> parsed_header_sources_str;
		//std::vector<std::string> parsed_header_names_str;
		//std::unordered_set<std::string> already_parsed_header_names;

		//std::deque<std::string> include_names_yet_to_parse(include_names.size());
		//std::copy(include_names.begin(), include_names.end(), include_names_yet_to_parse.begin());
		//while (!include_names_yet_to_parse.empty())
		//{
		//	std::string header_name = include_names_yet_to_parse.front();
		//	std::string header_path = locate_header_in_include_dirs(header_name, additional_include_directories);
		//	already_parsed_header_names.insert(header_name);

		//	// Allocating space for the incoming header
		//	std::vector<std::string> new_header_names;
		//	std::string header_source_code;
		//	read_source_code(header_path, header_source_code, &new_header_names);
		//	if (header_source_code.length() > 0)
		//	{
		//		parsed_header_names_str.push_back(header_name);
		//		parsed_header_sources_str.push_back(header_source_code);
		//	}

		//	// Adding the headers we just read to the queue of header files to read
		//	for (const std::string& new_header_name : new_header_names)
		//	{
		//		if (already_parsed_header_names.find(new_header_name) == already_parsed_header_names.end())
		//		{
		//			// Only pushing a new header to parse to the queue if it's not already 
		//			// in the queue (or has been parsed already)
		//			include_names_yet_to_parse.push_back(new_header_name);
		//			already_parsed_header_names.insert(new_header_name);
		//		}
		//	}

		//	include_names_yet_to_parse.pop_front();
		//}

		//std::vector<const char*> parsed_header_sources(parsed_header_sources_str.size());
		//std::vector<const char*> parsed_header_names(parsed_header_names_str.size());
		//std::transform(parsed_header_sources_str.begin(), parsed_header_sources_str.end(), parsed_header_sources.begin(), [](const std::string& s) {return s.c_str(); });
		//std::transform(parsed_header_names_str.begin(), parsed_header_names_str.end(), parsed_header_names.begin(), [](const std::string& s) {return s.c_str(); });

		// We recreating a vector of options here because we want to the additional_include_directories
		// as options. For that, we need an existing vector but we're not using the compiler_options options vector
		// passed in as a parameter because we don't want to modify the input and we're not sure it's
		// even a valid std::vector (it's an optional)
		int compiler_options_count = additional_include_directories.size() + (compiler_options ? compiler_options.value().size() : 0);
		std::vector<std::string> compiler_options_str;
		std::vector<const char*> compiler_options_cstr;
		compiler_options_str.reserve(compiler_options_count);
		compiler_options_cstr.reserve(compiler_options_count);
		if (compiler_options)
			std::copy(compiler_options.value().begin(), compiler_options.value().end(), compiler_options_str.begin());
		// Adding the additional include directories as options for the GPU compiler (-I flag before the include directory path)
		for (const std::string& additional_include_dir : additional_include_directories)
		{
			compiler_options_str.push_back("-I" + additional_include_dir);
			compiler_options_cstr.push_back(compiler_options_str.back().c_str());
		}

		return hiprtBuildTraceKernels(
			ctxt,
			1,
			&function_name,
			kernel_source_code.c_str(),
			kernel_file_path.c_str(),
			0, // parsed_header_names.size(),
			nullptr, // parsed_header_sources.data(),
			nullptr, // parsed_header_names.data(),
			compiler_options_cstr.size(),
			compiler_options_cstr.size() > 0 ? compiler_options_cstr.data() : nullptr,
			num_geom_types,
			num_ray_types,
			nullptr,
			&kernel_function_out,
			nullptr,
			use_compiler_cache);
	}
}
