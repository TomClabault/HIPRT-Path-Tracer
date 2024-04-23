/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

#include "GL/glew.h"

#include <string>
#include <vector>

class OpenGLUtils
{
public:
	static std::string add_macros_to_source(const std::string& source_string, const std::vector<std::string>& macros);
};

#endif