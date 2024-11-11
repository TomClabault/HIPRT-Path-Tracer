/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiTexture.h"

#include <Orochi/Orochi.h>

OrochiTexture::OrochiTexture(const Image8Bit& image, HIPfilter_mode filtering_mode)
{
	init_from_image(image, filtering_mode);
}

OrochiTexture::OrochiTexture(const Image32Bit& image, HIPfilter_mode filtering_mode)
{
	init_from_image(image, filtering_mode);
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

void OrochiTexture::operator=(OrochiTexture&& other) noexcept
{
	m_texture_array = std::move(other.m_texture_array);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
	other.m_texture_array = nullptr;
}

void OrochiTexture::init_from_image(const Image8Bit& image, HIPfilter_mode filtering_mode)
{
	int channels = image.channels;
	if (channels == 3)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "3-channels textures not supported on the GPU yet.");

		return;
	}

	width = image.width;
	height = image.height;

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	// The shenanigans with max(channels - 0/1/2/3) is to automatically set 0 or sizeof(unsigned char) * 8
	// bits in each channel depending on whether or not the input image indeed has that many
	// channels
	//
	// So if the input image only has 2 channels for example, then then Z and W channel will
	// be set to 0 bits by the 'channels - 2 > 0' and 'channels - 3 > 0' conditions respectively
	// which will be false
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(unsigned char) * 8 * (channels - 0 > 0),
																   sizeof(unsigned char) * 8 * (channels - 1 > 0),
																   sizeof(unsigned char) * 8 * (channels - 2 > 0),
																   sizeof(unsigned char) * 8 * (channels - 3 > 0),
																   hipChannelFormatKindUnsigned);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * channels * sizeof(unsigned char), image.width * sizeof(unsigned char) * channels, image.height, oroMemcpyHostToDevice));
	
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
	texture_descriptor.filterMode = filtering_mode;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
}

void OrochiTexture::init_from_image(const Image32Bit& image, HIPfilter_mode filtering_mode)
{
	int channels = image.channels;
	if (channels == 3)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "3-channels textures not supported on the GPU yet.");

		return;
	}

	width = image.width;
	height = image.height;

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	// The shenanigans with max(channels - 0/1/2/3) is to automatically set 0 or sizeof(float) * 8
	// bits in each channel depending on whether or not the input image indeed has that many
	// channels
	//
	// So if the input image only has 2 channels for example, then then Z and W channel will
	// be set to 0 bits by the 'channels - 2 > 0' and 'channels - 3 > 0' conditions respectively
	// which will be false
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(float) * 8 * (channels - 0 > 0),
																   sizeof(float) * 8 * (channels - 1 > 0),
																   sizeof(float) * 8 * (channels - 2 > 0),
																   sizeof(float) * 8 * (channels - 3 > 0),
																   hipChannelFormatKindUnsigned);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * channels * sizeof(float), image.width * sizeof(float) * channels, image.height, oroMemcpyHostToDevice));

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
	texture_descriptor.filterMode = filtering_mode;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}
