/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_SERIALIZATION_H
#define GPU_KERNEL_COMPILER_WINDOW_PROCESS_SERIALIZATION_H

#include "Compiler/GPUKernelCompilerWindowProcessCompilationRequest.h"

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

class GPUKernelCompilerWindowProcessSerialization
{
public:
	static bool write_request(const std::filesystem::path& request_file_path, const GPUKernelCompilerWindowProcessCompilationRequest& request);
	static bool read_request(const std::filesystem::path& request_file_path, GPUKernelCompilerWindowProcessCompilationRequest& request);

private:
	static constexpr unsigned int REQUEST_FILE_MAGIC		 = 0x47504B43;
	static constexpr unsigned int REQUEST_FILE_VERSION		 = 1;
	static constexpr unsigned int MAX_SERIALIZED_STRING_SIZE = 64 * 1024 * 1024;
	static constexpr unsigned int MAX_SERIALIZED_VECTOR_SIZE = 1 * 1024 * 1024;

	static bool write_unsigned_int(std::ofstream& output, unsigned int value);
	static bool read_unsigned_int(std::ifstream& input, unsigned int& value);
	static bool write_int(std::ofstream& output, int value);
	static bool read_int(std::ifstream& input, int& value);
	static bool write_bool(std::ofstream& output, bool value);
	static bool read_bool(std::ifstream& input, bool& value);
	static bool write_string(std::ofstream& output, const std::string& value);
	static bool read_string(std::ifstream& input, std::string& value);
	static bool write_string_vector(std::ofstream& output, const std::vector<std::string>& values);
	static bool read_string_vector(std::ifstream& input, std::vector<std::string>& values);
};

#endif // #ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_SERIALIZATION_H
