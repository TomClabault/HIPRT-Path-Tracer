/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPENGL_INTEROP_BUFFER_H
#define OPENGL_INTEROP_BUFFER_H

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "UI/DisplayView/DisplayTextureType.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "Orochi/Orochi.h"

extern ImGuiLogger g_imgui_logger;

template <typename T>
class OpenGLInteropBuffer
{
public:
	OpenGLInteropBuffer() {}
	OpenGLInteropBuffer(int element_count);
	~OpenGLInteropBuffer();

	GLuint get_opengl_buffer();

	void resize(int new_element_count);
	size_t get_element_count() const;
	size_t get_byte_size() const;

	/**
	 * Makes the buffer accesible to HIP/CUDA and returns the HIP/CUDA pointer
	 * to that buffer
	 */
	T* map();

	/**
	 * Makes the buffer accessible by OpenGL
	 */
	void unmap();

	/**
	 * Copies the buffer data to an OpenGL texture
	 */
	void unpack_to_GL_texture(GLuint texture, GLint texture_unit, int width, int height, DisplayTextureType texture_type);

	void free();

private:
	bool m_initialized = false;
	bool m_mapped = false;
	T* m_mapped_pointer = nullptr;

	size_t m_element_count = 0;

	GLuint m_buffer_name = -1;

	oroGraphicsResource_t m_buffer_resource = nullptr;
};

template <typename T>
OpenGLInteropBuffer<T>::OpenGLInteropBuffer(int element_count)
{
	glCreateBuffers(1, &m_buffer_name);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer_name);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);

	OROCHI_CHECK_ERROR(oroGraphicsGLRegisterBuffer(&m_buffer_resource, m_buffer_name, oroGraphicsRegisterFlagsNone));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_initialized = true;
	m_mapped = false;
	m_element_count = element_count;
}

template <typename T>
OpenGLInteropBuffer<T>::~OpenGLInteropBuffer()
{
	free();
}

template <typename T>
GLuint OpenGLInteropBuffer<T>::get_opengl_buffer()
{
	return m_buffer_name;
}

template <typename T>
void OpenGLInteropBuffer<T>::resize(int new_element_count)
{
	if (m_mapped)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to resize interop buffer while it is mapped! This is undefined behavior");

		return;
	}

	if (m_initialized)
	{
		oroGraphicsUnregisterResource(m_buffer_resource);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer_name);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, new_element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	}
	else
	{
		glCreateBuffers(1, &m_buffer_name);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer_name);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, new_element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	}

#ifndef OROCHI_ENABLE_CUEW
	// TODO hipGLGetDevices here is required for hipGraphicsGLRegisterBuffer to work. This is very scuffed.
	unsigned int count = 0;
	std::vector<int> devices(16);
	hipGLGetDevices(&count, devices.data(), 16, hipGLDeviceListAll);
#endif

	oroGraphicsGLRegisterBuffer(&m_buffer_resource, m_buffer_name, oroGraphicsRegisterFlagsNone);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_initialized = true;
	m_element_count = new_element_count;
}

template <typename T>
size_t OpenGLInteropBuffer<T>::get_element_count() const
{
	return m_element_count;
}

template <typename T>
size_t OpenGLInteropBuffer<T>::get_byte_size() const
{
	return m_element_count * sizeof(T);
}

template <typename T>
T* OpenGLInteropBuffer<T>::map()
{
	if (!m_initialized)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Mapping a buffer that hasn't been initialized!");

		Utils::debugbreak();

		return nullptr;
	}

	if (m_mapped)
		// Already mapped
		return m_mapped_pointer;

	size_t byte_size;
	OROCHI_CHECK_ERROR(oroGraphicsMapResources(1, reinterpret_cast<oroGraphicsResource_t*>(&m_buffer_resource), 0));
	OROCHI_CHECK_ERROR(oroGraphicsResourceGetMappedPointer((void**)(&m_mapped_pointer), &byte_size, reinterpret_cast<oroGraphicsResource_t>(m_buffer_resource)));

	m_mapped = true;
	return m_mapped_pointer;
}

template <typename T>
void OpenGLInteropBuffer<T>::unmap()
{
	if (!m_mapped)
		// Already unmapped
		return;

	OROCHI_CHECK_ERROR(oroGraphicsUnmapResources(1, reinterpret_cast<oroGraphicsResource_t*>(&m_buffer_resource), 0));

	m_mapped = false;
	m_mapped_pointer = nullptr;
}

template<typename T>
void OpenGLInteropBuffer<T>::unpack_to_GL_texture(GLuint texture, GLint texture_unit, int width, int height, DisplayTextureType texture_type)
{
	GLenum format = texture_type.get_gl_format();
	GLenum type = texture_type.get_gl_type();

	glActiveTexture(texture_unit);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, get_opengl_buffer());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

template<typename T>
void OpenGLInteropBuffer<T>::free()
{
	if (m_initialized)
	{
		glDeleteBuffers(1, &m_buffer_name);

		if (m_mapped)
			unmap();

		OROCHI_CHECK_ERROR(oroGraphicsUnregisterResource(reinterpret_cast<oroGraphicsResource_t>(m_buffer_resource)));
	}
	
	m_element_count = 0;
	m_initialized = false;
}

#endif
