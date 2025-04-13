/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "OpenGL/OpenGLProgram.h"
#include "UI/ImGui/ImGuiLogger.h"

extern ImGuiLogger g_imgui_logger;

OpenGLProgram::OpenGLProgram(const OpenGLShader& compiled_vertex)
{
	m_program = glCreateProgram();
	glAttachShader(m_program, compiled_vertex.get_shader());
	glLinkProgram(m_program);
}

OpenGLProgram::OpenGLProgram(const OpenGLShader& compiled_vertex, const OpenGLShader& compiled_fragment)
{
	m_program = glCreateProgram();
	glAttachShader(m_program, compiled_vertex.get_shader());
	glAttachShader(m_program, compiled_fragment.get_shader());
	glLinkProgram(m_program);
}

OpenGLProgram::~OpenGLProgram()
{
	glDeleteProgram(m_program);
}

void OpenGLProgram::attach(const OpenGLShader& compiled_shader)
{
	if (m_program == (unsigned int)(-1))
		m_program = glCreateProgram();

	glAttachShader(m_program, compiled_shader.get_shader());
	if (compiled_shader.get_shader_type() == OpenGLShader::COMPUTE_SHADER)
		m_is_compute = true;
}

void OpenGLProgram::link()
{
	glLinkProgram(m_program);

	if (m_is_compute)
		glGetProgramiv(m_program, GL_COMPUTE_WORK_GROUP_SIZE, m_compute_threads);
}

void OpenGLProgram::use()
{
	glUseProgram(m_program);
}

void OpenGLProgram::get_compute_threads(GLint threads[3])
{
	if (!m_is_compute)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "This program isn't a compute shader");

		return;
	}

	threads[0] = m_compute_threads[0];
	threads[1] = m_compute_threads[1];
	threads[2] = m_compute_threads[2];
}

void OpenGLProgram::set_uniform(const char* name, int value)
{
	glUniform1i(glGetUniformLocation(m_program, name), value);
}

void OpenGLProgram::set_uniform(const char* name, float value)
{
	glUniform1f(glGetUniformLocation(m_program, name), value);
}

void OpenGLProgram::set_uniform(const char* name, const float2& value)
{
	glUniform2f(glGetUniformLocation(m_program, name), value.x, value.y);
}

void OpenGLProgram::set_uniform(const char* name, const float3& value)
{
	glUniform3f(glGetUniformLocation(m_program, name), value.x, value.y, value.z);
}

void OpenGLProgram::set_uniform(const char* name, int count, const float* values)
{
	glUniform3fv(glGetUniformLocation(m_program, name), count, values);
}

void OpenGLProgram::set_uniform(const char* name, const float4& value)
{
	glUniform4f(glGetUniformLocation(m_program, name), value.x, value.y, value.z, value.w);
}

