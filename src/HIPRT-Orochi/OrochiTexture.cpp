/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiTexture.h"

#include <Orochi/Orochi.h>

OrochiTexture::OrochiTexture(const ImageRGBA32F& image)
{
	init_from_image(image);
}

OrochiTexture::OrochiTexture(OrochiTexture&& other)
{
	m_texture_array = std::move(other.m_texture_array);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
	other.m_texture_array = nullptr;
}

OrochiTexture::~OrochiTexture()
{
	if (m_texture)
		oroDestroyTextureObject(m_texture);

	if (m_texture_array)
		oroFree(m_texture_array);
}

void OrochiTexture::operator=(OrochiTexture&& other)
{
	m_texture_array = std::move(other.m_texture_array);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
	other.m_texture_array = nullptr;
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}

oroTextureObject_t* OrochiTexture::get_device_texture_pointer()
{
	return &m_texture;
}
