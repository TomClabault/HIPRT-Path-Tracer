/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Utils/opengl_utils.h"
#include "Utils/utils.h"

#include <stdexcept>

std::string OpenGLUtils::add_macros_to_source(const std::string& source_string, const std::vector<std::string>& macros)
{
	std::string modified_source = source_string;

	size_t version_pos = modified_source.find("#version");
	if (version_pos != std::string::npos)
	{	
		size_t line_return_pos = modified_source.find('\n', version_pos);
		size_t after_return = line_return_pos + 1;

		for (const std::string& macro : macros)
			modified_source = modified_source.insert(after_return, macro + "\n");
	}
	else
		throw new std::runtime_error("No #version directive found in shader...");

	return modified_source;
}
