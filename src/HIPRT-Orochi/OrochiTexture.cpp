/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiTexture.h"

#include <Orochi/Orochi.h>

OrochiTexture::OrochiTexture(const Image8Bit& image)
{
	init_from_image(image);
}

OrochiTexture::OrochiTexture(const Image32Bit& image)
{
	init_from_image(image);
}

OrochiTexture::OrochiTexture(OrochiTexture&& other) noexcept
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

void OrochiTexture::init_from_image(const Image8Bit& image)
{
	if (image.channels == 3)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "3-channels textures not supported on the GPU yet.");

		return;
	}

	width = image.width;
	height = image.height;

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(unsigned char) * 8, sizeof(unsigned char) * 8, sizeof(unsigned char) * 8, sizeof(unsigned char) * 8, oroChannelFormatKindUnsigned);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * image.channels * sizeof(unsigned char), image.width * sizeof(unsigned char) * image.channels, image.height, oroMemcpyHostToDevice));

	// Resource descriptor
	ORO_RESOURCE_DESC resource_descriptor;
	std::memset(&resource_descriptor, 0, sizeof(resource_descriptor));
	resource_descriptor.resType = ORO_RESOURCE_TYPE_ARRAY;
	resource_descriptor.res.array.hArray = m_texture_array;

	ORO_TEXTURE_DESC texture_descriptor;
	std::memset(&texture_descriptor, 0, sizeof(texture_descriptor));
	texture_descriptor.addressMode[0] = ORO_TR_ADDRESS_MODE_WRAP;
	texture_descriptor.addressMode[1] = ORO_TR_ADDRESS_MODE_WRAP;
	texture_descriptor.addressMode[2] = ORO_TR_ADDRESS_MODE_WRAP;
	texture_descriptor.filterMode = ORO_TR_FILTER_MODE_LINEAR;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
}

void OrochiTexture::init_from_image(const Image32Bit& image)
{
	if (image.channels == 1 || image.channels == 3)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "1-channel & 3-channels textures not supported on the GPU yet.");

		return;
	}

	width = image.width;
	height = image.height;

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, oroChannelFormatKindFloat);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * image.channels * sizeof(float), image.width * sizeof(float) * image.channels, image.height, oroMemcpyHostToDevice));

	// Resource descriptor
	ORO_RESOURCE_DESC resource_descriptor;
	std::memset(&resource_descriptor, 0, sizeof(resource_descriptor));
	resource_descriptor.resType = ORO_RESOURCE_TYPE_ARRAY;
	resource_descriptor.res.array.hArray = m_texture_array;

	ORO_TEXTURE_DESC texture_descriptor;
	std::memset(&texture_descriptor, 0, sizeof(texture_descriptor));
	texture_descriptor.addressMode[0] = ORO_TR_ADDRESS_MODE_WRAP;
	texture_descriptor.addressMode[1] = ORO_TR_ADDRESS_MODE_WRAP;
	texture_descriptor.addressMode[2] = ORO_TR_ADDRESS_MODE_WRAP;
	texture_descriptor.filterMode = ORO_TR_FILTER_MODE_LINEAR;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}
