/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerWindowProcessSerialization.h"

#include <fstream>
#include <limits>

bool GPUKernelCompilerWindowProcessSerialization::write_unsigned_int(std::ofstream& output, unsigned int value)
{
	output.write(reinterpret_cast<const char*>(&value), sizeof(value));
	return output.good();
}

bool GPUKernelCompilerWindowProcessSerialization::read_unsigned_int(std::ifstream& input, unsigned int& value)
{
	input.read(reinterpret_cast<char*>(&value), sizeof(value));
	return input.good();
}

bool GPUKernelCompilerWindowProcessSerialization::write_int(std::ofstream& output, int value)
{
	output.write(reinterpret_cast<const char*>(&value), sizeof(value));
	return output.good();
}

bool GPUKernelCompilerWindowProcessSerialization::read_int(std::ifstream& input, int& value)
{
	input.read(reinterpret_cast<char*>(&value), sizeof(value));
	return input.good();
}

bool GPUKernelCompilerWindowProcessSerialization::write_bool(std::ofstream& output, bool value)
{
	unsigned int serialized_value = value ? 1 : 0;
	return write_unsigned_int(output, serialized_value);
}

bool GPUKernelCompilerWindowProcessSerialization::read_bool(std::ifstream& input, bool& value)
{
	unsigned int serialized_value = 0;
	if (!read_unsigned_int(input, serialized_value) || serialized_value > 1)
		return false;

	value = serialized_value == 1;
	return true;
}

bool GPUKernelCompilerWindowProcessSerialization::write_string(std::ofstream& output, const std::string& value)
{
	if (value.size() > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
		return false;

	unsigned int value_size = static_cast<unsigned int>(value.size());
	if (!write_unsigned_int(output, value_size))
		return false;

	if (value_size > 0)
		output.write(value.data(), value_size);

	return output.good();
}

bool GPUKernelCompilerWindowProcessSerialization::read_string(std::ifstream& input, std::string& value)
{
	unsigned int value_size = 0;
	if (!read_unsigned_int(input, value_size) || value_size > MAX_SERIALIZED_STRING_SIZE)
		return false;

	value.resize(value_size);
	if (value_size > 0)
		input.read(value.data(), value_size);

	return input.good();
}

bool GPUKernelCompilerWindowProcessSerialization::write_string_vector(std::ofstream& output, const std::vector<std::string>& values)
{
	if (values.size() > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
		return false;

	unsigned int value_count = static_cast<unsigned int>(values.size());
	if (!write_unsigned_int(output, value_count))
		return false;

	for (const std::string& value : values)
		if (!write_string(output, value))
			return false;

	return true;
}

bool GPUKernelCompilerWindowProcessSerialization::read_string_vector(std::ifstream& input, std::vector<std::string>& values)
{
	unsigned int value_count = 0;
	if (!read_unsigned_int(input, value_count) || value_count > MAX_SERIALIZED_VECTOR_SIZE)
		return false;

	values.resize(value_count);
	for (std::string& value : values)
		if (!read_string(input, value))
			return false;

	return true;
}

bool GPUKernelCompilerWindowProcessSerialization::write_request(const std::filesystem::path& request_file_path,
																const GPUKernelCompilerWindowProcessCompilationRequest& request)
{
	std::ofstream output(request_file_path, std::ios::binary | std::ios::trunc);
	if (!output.is_open())
		return false;

	if (!write_unsigned_int(output, REQUEST_FILE_MAGIC) || !write_unsigned_int(output, REQUEST_FILE_VERSION) ||
		!write_string(output, request.kernel_file_path) || !write_string(output, request.kernel_function_name) ||
		!write_string_vector(output, request.additional_include_directories) || !write_string_vector(output, request.compiler_options) ||
		!write_int(output, request.num_geom_types) || !write_int(output, request.num_ray_types) || !write_bool(output, request.use_compiler_cache) ||
		!write_bool(output, request.has_function_name_sets))
		return false;

	if (request.function_name_sets.size() > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
		return false;

	unsigned int function_name_set_count = static_cast<unsigned int>(request.function_name_sets.size());
	if (!write_unsigned_int(output, function_name_set_count))
		return false;

	for (const GPUKernelCompilerWindowProcessFunctionNameSet& function_name_set : request.function_name_sets)
	{
		bool has_intersect_function_name = function_name_set.intersect_function_name.has_value();
		bool has_filter_function_name	 = function_name_set.filter_function_name.has_value();
		if (!write_bool(output, has_intersect_function_name) || !write_bool(output, has_filter_function_name))
			return false;

		if (has_intersect_function_name && !write_string(output, function_name_set.intersect_function_name.value()))
			return false;
		if (has_filter_function_name && !write_string(output, function_name_set.filter_function_name.value()))
			return false;
	}

	if (!write_string(output, request.additional_cache_key) || !write_int(output, request.device_index))
		return false;

	output.flush();
	return output.good();
}

bool GPUKernelCompilerWindowProcessSerialization::read_request(const std::filesystem::path& request_file_path,
															   GPUKernelCompilerWindowProcessCompilationRequest& request)
{
	std::ifstream input(request_file_path, std::ios::binary);
	if (!input.is_open())
		return false;

	unsigned int magic	 = 0;
	unsigned int version = 0;
	if (!read_unsigned_int(input, magic) || !read_unsigned_int(input, version) || magic != REQUEST_FILE_MAGIC || version != REQUEST_FILE_VERSION ||
		!read_string(input, request.kernel_file_path) || !read_string(input, request.kernel_function_name) ||
		!read_string_vector(input, request.additional_include_directories) || !read_string_vector(input, request.compiler_options) ||
		!read_int(input, request.num_geom_types) || !read_int(input, request.num_ray_types) || !read_bool(input, request.use_compiler_cache) ||
		!read_bool(input, request.has_function_name_sets))
		return false;

	unsigned int function_name_set_count = 0;
	if (!read_unsigned_int(input, function_name_set_count) || function_name_set_count > MAX_SERIALIZED_VECTOR_SIZE)
		return false;

	request.function_name_sets.resize(function_name_set_count);
	for (GPUKernelCompilerWindowProcessFunctionNameSet& function_name_set : request.function_name_sets)
	{
		bool has_intersect_function_name = false;
		bool has_filter_function_name	 = false;
		if (!read_bool(input, has_intersect_function_name) || !read_bool(input, has_filter_function_name))
			return false;

		if (has_intersect_function_name)
		{
			std::string intersect_function_name;
			if (!read_string(input, intersect_function_name))
				return false;
			function_name_set.intersect_function_name = intersect_function_name;
		}

		if (has_filter_function_name)
		{
			std::string filter_function_name;
			if (!read_string(input, filter_function_name))
				return false;
			function_name_set.filter_function_name = filter_function_name;
		}
	}

	return read_string(input, request.additional_cache_key) && read_int(input, request.device_index);
}
