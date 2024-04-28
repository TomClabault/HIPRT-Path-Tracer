/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPENGL_INTEROP_BUFFER_H
#define OPENGL_INTEROP_BUFFER_H

#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "Orochi/Orochi.h"
#include "UI/DisplayTextureType.h"

// TODO this class uses HIP for the registering / mapping because Orochi doesn't have opengl interop yet ?
// we should be using Orochi here instead of HIP because this is not NVIDIA friendly since we would have to
// link with HIP during the compilation. That's why NVIDIA OpenGL interop is disabled for now
template <typename T>
class OpenGLInteropBuffer
{
public:
	OpenGLInteropBuffer() {}
	OpenGLInteropBuffer(int element_count);
	~OpenGLInteropBuffer();

	GLuint get_opengl_buffer();

	void resize(int new_element_count);

	/*
	 * This function is stricly an alias for map()
	 */
	T* get_device_pointer();
	T* map();
	void unmap();

	void unpack_to_texture(GLuint texture, GLint texture_unit, int width, int height, DisplayTextureType texture_type);

private:
	bool m_initialized = false;
	bool m_mapped = false;
	T* m_mapped_pointer;

	size_t m_byte_size = 0;

	GLuint m_buffer_name = -1;
	oroGraphicsResource_t m_buffer_resource;
};

template <typename T>
OpenGLInteropBuffer<T>::OpenGLInteropBuffer(int element_count)
{
	glCreateBuffers(1, &m_buffer_name);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer_name);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	hipGraphicsGLRegisterBuffer((hipGraphicsResource_t*)&m_buffer_resource, m_buffer_name, hipGraphicsRegisterFlagsNone);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_initialized = true;
	m_mapped = false;
	m_byte_size = element_count * sizeof(T);
}

template <typename T>
GLuint OpenGLInteropBuffer<T>::get_opengl_buffer()
{
	return m_buffer_name;
}

template <typename T>
void OpenGLInteropBuffer<T>::resize(int new_element_count)
{
	if (m_initialized)
	{
		hipGraphicsUnregisterResource(m_buffer_resource);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer_name);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, new_element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	}
	else
	{
		glCreateBuffers(1, &m_buffer_name);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_buffer_name);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, new_element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	}

	// TODO hipGLGetDevices here is required for hipGraphicsGLRegisterBuffer to work. This is very scuffed.
	unsigned int count = 0;
	std::vector<int> devices(16);
	hipGLGetDevices(&count, devices.data(), 16, hipGLDeviceListAll);
	OROCHI_CHECK_ERROR(hipGraphicsGLRegisterBuffer((hipGraphicsResource_t*)&m_buffer_resource, m_buffer_name, hipGraphicsRegisterFlagsNone));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_initialized = true;
	m_byte_size = new_element_count * sizeof(T);
}

template <typename T>
T* OpenGLInteropBuffer<T>::map()
{
	if (!m_initialized)
		return nullptr;

	if (m_mapped)
		// Already mapped
		return m_mapped_pointer;

	OROCHI_CHECK_ERROR(oroGraphicsMapResources(1, &m_buffer_resource, 0));
	OROCHI_CHECK_ERROR(oroGraphicsResourceGetMappedPointer((void**)&m_mapped_pointer, &m_byte_size, m_buffer_resource));

	m_mapped = true;
	return m_mapped_pointer;
}

template <typename T>
T* OpenGLInteropBuffer<T>::get_device_pointer()
{
	return map();
}

template <typename T>
void OpenGLInteropBuffer<T>::unmap()
{
	if (!m_mapped)
		// Already unmapped
		return;

	OROCHI_CHECK_ERROR(oroGraphicsUnmapResources(1, &m_buffer_resource, 0));

	m_mapped = false;
	m_mapped_pointer = nullptr;
}

template<typename T>
void OpenGLInteropBuffer<T>::unpack_to_texture(GLuint texture, GLint texture_unit, int width, int height, DisplayTextureType texture_type)
{
	GLenum format = texture_type.get_gl_format();
	GLenum type = texture_type.get_gl_type();

	glActiveTexture(texture_unit);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, get_opengl_buffer());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

template <typename T>
OpenGLInteropBuffer<T>::~OpenGLInteropBuffer()
{
	if (m_initialized)
	{
		glDeleteBuffers(1, &m_buffer_name);

		if (m_mapped)
			unmap();

		OROCHI_CHECK_ERROR(oroGraphicsUnregisterResource(m_buffer_resource));
	}
}

#endif
