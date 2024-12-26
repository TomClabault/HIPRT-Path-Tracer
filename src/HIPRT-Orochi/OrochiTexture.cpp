/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiTexture.h"

#include <Orochi/Orochi.h>

OrochiTexture::OrochiTexture(const Image8Bit& image, hipTextureFilterMode filtering_mode, hipTextureAddressMode address_mode)
{
	init_from_image(image, filtering_mode, address_mode);
}

OrochiTexture::OrochiTexture(const Image32Bit& image, hipTextureFilterMode filtering_mode, hipTextureAddressMode address_mode)
{
	init_from_image(image, filtering_mode, address_mode);
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

void create_texture_from_array_cuda(void* m_texture_array, void* m_texture, void* filtering_mode, void* address_mode, bool read_mode_float_normalized);

void OrochiTexture::create_texture_from_array(hipTextureFilterMode filtering_mode, hipTextureAddressMode address_mode, bool read_mode_float_normalized)
{
#ifndef OROCHI_ENABLE_CUEW
	// Using native HIP here to access 'normalizedCoords' which isn't  exposed by Orochi

	hipResourceDesc resource_descriptor = {};
	resource_descriptor.resType = hipResourceTypeArray;
	resource_descriptor.res.array.array = m_texture_array;

	hipTextureDesc texture_descriptor = {};
	texture_descriptor.addressMode[0] = address_mode;
	texture_descriptor.addressMode[1] = address_mode;
	texture_descriptor.addressMode[2] = address_mode;
	texture_descriptor.filterMode = filtering_mode;
	texture_descriptor.normalizedCoords = true;
	texture_descriptor.readMode = read_mode_float_normalized ? hipTextureReadMode::hipReadModeNormalizedFloat : hipTextureReadMode::hipReadModeElementType;
	texture_descriptor.sRGB = false;

	OROCHI_CHECK_ERROR(hipCreateTextureObject(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
#else
	// Using native CUDA here to access 'normalizedCoords' which isn't  exposed by Orochi
	// Note that this function is defined in another compile unit because we need to include CUDA headers
	// and they conflict with HIP headers (structures redefinition, float2, float4, ...) it seems so we need to separate them
	create_texture_from_array_cuda(m_texture_array, &m_texture, &filtering_mode, &address_mode, read_mode_float_normalized);
#endif
}

void OrochiTexture::init_from_image(const Image8Bit& image, hipTextureFilterMode filtering_mode, hipTextureAddressMode address_mode)
{
	int channels = image.channels;
	if (channels == 3 || channels > 4)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "3-channels textures not supported on the GPU yet.");

		return;
	}

	width = image.width;
	height = image.height;

	int bits_channel_x = (channels >= 1) ? 8 : 0; // First channel (e.g., Red)
	int bits_channel_y = (channels >= 2) ? 8 : 0; // Second channel (e.g., Green)
	int bits_channel_z = (channels >= 3) ? 8 : 0; // Third channel (e.g., Blue)
	int bits_channel_w = (channels == 4) ? 8 : 0; // Fourth channel (e.g., Alpha)
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(bits_channel_x, bits_channel_y, bits_channel_z, bits_channel_w,
		oroChannelFormatKindUnsigned);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), 
		image.width * channels * sizeof(unsigned char), 
		image.width * sizeof(unsigned char) * channels, 
		image.height, oroMemcpyHostToDevice));
	
	create_texture_from_array(filtering_mode, address_mode, true);
}

void OrochiTexture::init_from_image(const Image32Bit& image, hipTextureFilterMode filtering_mode, hipTextureAddressMode address_mode)
{
	int channels = image.channels;
	if (channels == 3 || channels > 4)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "3-channels textures not supported on the GPU yet.");
		
		return;
	}

	width = image.width;
	height = image.height;

	if (width == 0 || height == 0)
	{
		std::cerr << "Image given to OrochiTexture is 0 in width or height" << std::endl;
		Utils::debugbreak();
	}

	int bits_channel_x = (channels >= 1) ? 32 : 0; // First channel (e.g., Red)
	int bits_channel_y = (channels >= 2) ? 32 : 0; // Second channel (e.g., Green)
	int bits_channel_z = (channels >= 3) ? 32 : 0; // Third channel (e.g., Blue)
	int bits_channel_w = (channels == 4) ? 32 : 0; // Fourth channel (e.g., Alpha)
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(bits_channel_x, bits_channel_y, bits_channel_z, bits_channel_w,
																   oroChannelFormatKindFloat);

	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), 
		image.width * channels * sizeof(float), 
		image.width * sizeof(float) * channels, 
		image.height, oroMemcpyHostToDevice));

	create_texture_from_array(filtering_mode, address_mode, false);
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}
