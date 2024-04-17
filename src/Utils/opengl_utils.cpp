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

GLuint OpenGLUtils::compile_shader_program(const std::string& vertex_shader_file_path, const std::string& fragment_shader_file_path)
{
	std::string vertex_string = Utils::file_to_string(vertex_shader_file_path.c_str());
	const char* vertex_shader_text = vertex_string.c_str();

	// Tone mapping fragment shader
	std::string frag_string = Utils::file_to_string(GLSL_SHADERS_DIRECTORY "/display.frag");
	const char* fragment_shader_text = frag_string.c_str();

	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
	glCompileShader(vertex_shader);
	if (!print_shader_compile_error(vertex_shader))
		throw new std::runtime_error("Unable to compile vertex shader given at this path: " + vertex_shader_file_path);

	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
	glCompileShader(fragment_shader);
	if (!print_shader_compile_error(fragment_shader))
		throw new std::runtime_error("Unable to compile fragment shader given at this path: " + fragment_shader_file_path);

	GLuint display_program = glCreateProgram();
	glAttachShader(display_program, vertex_shader);
	glAttachShader(display_program, fragment_shader);
	glLinkProgram(display_program);

	return display_program;
}

GLuint OpenGLUtils::compile_computer_program(const std::string& compute_shader_file_path)
{
	std::string compute_string = Utils::file_to_string(compute_shader_file_path.c_str());
	std::vector<std::string> macros;
	macros.push_back("#define COMPUTE_SHADER");
	compute_string = OpenGLUtils::add_macros_to_source(compute_string, macros);
	const char* compute_shader_text = compute_string.c_str();

	GLuint compute_shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(compute_shader, 1, &compute_shader_text, NULL);
	glCompileShader(compute_shader);
	if (!OpenGLUtils::print_shader_compile_error(compute_shader))
		throw new std::runtime_error("Unable to compile compute shader given at this path: " + compute_shader_file_path);

	GLuint display_program = glCreateProgram();
	glAttachShader(display_program, compute_shader);
	glLinkProgram(display_program);

	return display_program;
}

bool OpenGLUtils::print_shader_compile_error(GLuint shader)
{
	GLint isCompiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

		// The maxLength includes the NULL character
		std::vector<GLchar> errorLog(maxLength);
		glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

		std::cout << errorLog.data() << std::endl;

		// Provide the infolog in whatever manor you deem best.
		// Exit with failure.
		glDeleteShader(shader); // Don't leak the shader.

		return false;
	}

	return true;
}