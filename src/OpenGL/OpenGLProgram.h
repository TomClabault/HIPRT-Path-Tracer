/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPENGL_PROGRAM_H
#define OPENGL_PROGRAM_H

#include "gl/glew.h"
#include "HostDeviceCommon/math.h"
#include "OpenGL/OpenGLShader.h"

class OpenGLProgram
{
public:
	OpenGLProgram() : m_program(-1) { }
	OpenGLProgram(OpenGLProgram& other) = delete;
	OpenGLProgram(const OpenGLShader& vertex);
	OpenGLProgram(const OpenGLShader& compiled_vertex, const OpenGLShader& compiled_fragment);
	~OpenGLProgram();

	void attach(const OpenGLShader& compiled_shader);
	void link();
	void use();

	void get_compute_threads(GLint threads[3]);

	void set_uniform(const char* name, int value);
	void set_uniform(const char* name, float value);
	void set_uniform(const char* name, const float2& value);
	void set_uniform(const char* name, const float3& value);
	void set_uniform(const char* name, int count, const float* values);
	void set_uniform(const char* name, const float4& value);

private:

	bool m_is_compute = false;
	GLuint m_program = -1;
	GLint m_compute_threads[3];
};

#endif