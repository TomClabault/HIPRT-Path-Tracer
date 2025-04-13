/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPENGL_SHADER_H
#define OPENGL_SHADER_H

#include "GL/glew.h"

#include <string>
#include <vector>

class OpenGLShader
{
public:
	enum ShaderType
	{
		UNDEFINED,
		VERTEX_SHADER = GL_VERTEX_SHADER,
		FRAGMENT_SHADER = GL_FRAGMENT_SHADER,
		COMPUTE_SHADER = GL_COMPUTE_SHADER
	};

	OpenGLShader() : m_compiled_shader(-1), m_shader_type(ShaderType::UNDEFINED) {}
	/**
	 * If given, macros should be entire strings (with the '#') like "#define MY_MACRO", #if X, #endif, ... 
	 */
	OpenGLShader(const std::string& source_code, ShaderType type, const std::vector<std::string>& macros = std::vector<std::string>());
	OpenGLShader(const char* filepath, ShaderType type, const std::vector<std::string>& macros = std::vector<std::string>());

	std::string& get_source();
	const std::string& get_source() const;

	bool has_filepath() const;
	std::string& get_path();
	const std::string& get_path() const;

	void set_source(const std::string& source_code);
	void set_source_from_file(const char* filepath);

	GLuint get_shader() const;
	ShaderType get_shader_type() const;

	void compile(const std::vector<std::string>& macros);
	std::string add_macros_to_source(const std::vector<std::string>& macros);

	static bool print_shader_compile_error(GLuint shader);

private:
	std::string m_filepath;
	std::string m_source_code;

	ShaderType m_shader_type;
	GLuint m_compiled_shader;
};

#endif