/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "OpenGL/OpenGLShader.h"
#include "Utils/utils.h"

OpenGLShader::OpenGLShader(const std::string& source_code, ShaderType type)
{
	m_shader_type = type;

	set_source(source_code);
	compile();
}

OpenGLShader::OpenGLShader(const char* filepath, ShaderType type)
{
	m_shader_type = type;

	set_source_from_file(filepath);
	compile();
}

std::string& OpenGLShader::get_source()
{
	return m_source_code;
}

const std::string& OpenGLShader::get_source() const
{
	return m_source_code;
}

bool OpenGLShader::has_filepath() const
{
	return m_filepath.length() > 0;
}

std::string& OpenGLShader::get_path()
{
	return m_filepath;
}

const std::string& OpenGLShader::get_path() const
{
	return m_filepath;
}

void OpenGLShader::set_source(const std::string& source_code)
{
	m_source_code = source_code;
}

void OpenGLShader::set_source_from_file(const char* filepath)
{
	m_source_code = Utils::file_to_string(filepath);
}

GLuint OpenGLShader::get_shader() const
{
	return m_compiled_shader;
}

OpenGLShader::ShaderType OpenGLShader::get_shader_type() const
{
	return m_shader_type;
}

void OpenGLShader::compile()
{
	const char* vertex_shader_text = m_source_code.c_str();

	m_compiled_shader = glCreateShader(m_shader_type);
	glShaderSource(m_compiled_shader, 1, &vertex_shader_text, NULL);
	glCompileShader(m_compiled_shader);
	if (!print_shader_compile_error(m_compiled_shader))
	{
		if (has_filepath())
			throw new std::runtime_error("Unable to compile vertex shader given at this path: " + get_path());
		else
			throw new std::runtime_error("Unable to compile vertex shader");
	}
}

bool OpenGLShader::print_shader_compile_error(GLuint shader)
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
