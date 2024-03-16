#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

#include "GL/glew.h"

#include <string>
#include <vector>

class OpenGLUtils
{
public:
	static std::string add_macros_to_source(const std::string& source_string, const std::vector<std::string>& macros);

	static GLuint compile_shader_program(const std::string& vertex_shader_file_path, const std::string& fragment_shader_file_path);
	static GLuint compile_computer_program(const std::string& compute_shader_file_path);

	static bool print_shader_compile_error(GLuint shader);
};

#endif