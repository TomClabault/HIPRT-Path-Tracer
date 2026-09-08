/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_TEXTURE_TYPE_H
#define DISPLAY_TEXTURE_TYPE_H

#include "GL/glew.h"

class DisplayTextureType
{
public:
	enum Value
	{
		UNINITIALIZED,
		FLOAT3
	};

	constexpr DisplayTextureType() : m_value(Value::FLOAT3) {}
	constexpr DisplayTextureType(Value val) : m_value(val) {}

	GLint get_gl_internal_format()
	{
		switch (m_value)
		{
		case DisplayTextureType::FLOAT3:
			return GL_RGB32F;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	GLenum get_gl_format()
	{
		switch (m_value)
		{
		case DisplayTextureType::FLOAT3:
			return GL_RGB;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	GLenum get_gl_type()
	{
		switch (m_value)
		{
		case DisplayTextureType::FLOAT3:
			return GL_FLOAT;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	size_t sizeof_type()
	{
		switch (m_value)
		{
		case DisplayTextureType::FLOAT3:
			return sizeof(float) * 3;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	bool operator==(const DisplayTextureType& other)
	{
		return m_value == other.m_value;
	}
	bool operator!=(const DisplayTextureType& other)
	{
		return m_value != other.m_value;
	}

private:
	Value m_value;
};

#endif // #ifndef DISPLAY_TEXTURE_TYPE_H
