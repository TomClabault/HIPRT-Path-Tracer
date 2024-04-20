/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GL/glew.h"
#include "Orochi/Orochi.h"

template <typename T>
class OpenGLInteropBuffer
{
public:
	OpenGLInteropBuffer() {}
	OpenGLInteropBuffer(int element_count);
	~OpenGLInteropBuffer();

	void resize(int new_element_count);

	T* map();
	void unmap();

private:
	bool m_initialized = false;
	unsigned int m_byte_size = 0;

	GLuint m_buffer_name = -1;
	oroGraphicsResource_t m_mapped_buffer_resouce;
};

template <typename T>
OpenGLInteropBuffer<T>::OpenGLInteropBuffer(int element_count)
{
	glCreateBuffers(1, &m_buffer_name);
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer_name);
	glBufferData(GL_ARRAY_BUFFER, element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_initialized = true;
	m_byte_size = element_count * sizeof(T);
}

template <typename T>
void OpenGLInteropBuffer<T>::resize(int new_element_count)
{
	if (m_initialized)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_buffer_name);
		glBufferData(GL_ARRAY_BUFFER, new_element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	}
	else
	{
		glCreateBuffers(1, &m_buffer_name);
		glBindBuffer(GL_ARRAY_BUFFER, m_buffer_name);
		glBufferData(GL_ARRAY_BUFFER, new_element_count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);

		m_initialized = true;
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_byte_size = new_element_count * sizeof(T);
}

template <typename T>
T* OpenGLInteropBuffer<T>::map()
{
	if (!m_initialized)
		return nullptr;

	T* mapped_pointer;
	oroGraphicsMapResources(1, &m_mapped_buffer_resouce, 0);
	oroGraphicsResourceGetMappedPointer(&mapped_pointer, &m_byte_size, m_mapped_buffer_resouce);

	return mapped_pointer;
}

template <typename T>
void OpenGLInteropBuffer<T>::unmap()
{
	oroGraphicsUnmapResources(1, &m_mapped_buffer_resouce, 0);
}

template <typename T>
OpenGLInteropBuffer<T>::~OpenGLInteropBuffer()
{
	if (m_initialized)
		glDeleteBuffers(1, &m_buffer_name);
}
