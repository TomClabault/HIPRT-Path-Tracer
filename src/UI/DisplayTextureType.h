/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DISPLAY_TEXTURE_TYPE_H
#define DISPLAY_TEXTURE_TYPE_H

#include "gl/glew.h"

class DisplayTextureType
{
public:
	enum Value
	{
		UNINITIALIZED,
		FLOAT3,
		INT
	};

	constexpr DisplayTextureType() : value(Value::FLOAT3) { }
	constexpr DisplayTextureType(Value val) : value(val) { }

	GLint get_gl_internal_format()
	{
		switch (value)
		{
		case DisplayTextureType::FLOAT3:
			return GL_RGB32F;

		case DisplayTextureType::INT:
			return GL_R32I;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	GLenum get_gl_format()
	{
		switch (value)
		{
		case DisplayTextureType::FLOAT3:
			return GL_RGB;

		case DisplayTextureType::INT:
			return GL_RED_INTEGER;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	GLenum get_gl_type()
	{
		switch (value)
		{
		case DisplayTextureType::FLOAT3:
			return GL_FLOAT;

		case DisplayTextureType::INT:
			return GL_INT;

		default:
			throw std::runtime_error("Invalid value of DisplayTextureType");
		}
	}

	bool operator !=(const DisplayTextureType& other) { return value != other.value; }

private:
	Value value;
};

#endif
