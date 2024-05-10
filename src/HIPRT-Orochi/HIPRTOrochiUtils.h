/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRTPT_OROCHI_UTILS_H
#define HIPRTPT_OROCHI_UTILS_H

#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>

#include <filesystem>
#include <optional>

#define OROCHI_CHECK_ERROR( error ) ( orochi_check_error( error, __FILE__, __LINE__ ) )
#define OROCHI_RTC_CHECK_ERROR( error ) ( orochi_rtc_check_error( error, __FILE__, __LINE__ ) )
#define HIPRT_CHECK_ERROR( error ) ( hiprt_check_error( error, __FILE__, __LINE__ ) )

// This flag isn't defined in Orochi for some reasons ?
// It allows sampling textures with normalized coordinates in [0, 1[ instead of 
// [0, width[
#define ORO_TRSF_NORMALIZED_COORDINATES 0x02

namespace HIPPTOrochiUtils
{
	/*
	 * This function looks for a file that has the name 'header_name'
	 * in the list of directories given by 'include_directories'
	 * 
	 * If the file is found, the path to the file is returned by concatenating
	 * the include directory in which it was found with the file name
	 * 
	 * The empty string is returned if the file is found in none of the include
	 * directories given
	 */
	std::string locate_header_in_include_dirs(const std::string& header_name, const std::vector<std::string> include_directories);

	/*
	 * Reads a given file, outputs its code in 'sourceCode' and a list of the names
	 * of the files included in the source file by #include directives in 'includes'
	 * 
	 * If 'includes' is nullptr, then no include names will be returned
	 */
	bool read_source_code(const std::string& path, std::string& sourceCode, std::vector<std::string>* includes);

	hiprtError build_trace_kernel(hiprtContext ctxt,
		const std::string& kernel_file_path,
		const char* function_name,
		hiprtApiFunction& kernel_function_out,
		const std::vector<std::string>& additional_include_directories,
		const std::optional<std::vector<const char*>>& compiler_options,
		unsigned int num_geom_types, unsigned int num_ray_types,
		bool use_compiler_cache);
}

void orochi_check_error(oroError res, const char* file, uint32_t line);
void orochi_rtc_check_error(orortcResult res, const char* file, uint32_t line);
void hiprt_check_error(hiprtError res, const char* file, uint32_t line);

#endif
