/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiTexture3D.h"

#include <Orochi/Orochi.h>

OrochiTexture3D::OrochiTexture3D(const std::vector<Image8Bit>& images, HIPfilter_mode filtering_mode)
{
	init_from_images(images, filtering_mode);
}

OrochiTexture3D::OrochiTexture3D(const std::vector<Image32Bit>& images, HIPfilter_mode filtering_mode)
{
	init_from_images(images, filtering_mode);
}

OrochiTexture3D::OrochiTexture3D(OrochiTexture3D&& other) noexcept
{
	m_texture_array = std::move(other.m_texture_array);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
	other.m_texture_array = nullptr;
}

OrochiTexture3D::~OrochiTexture3D()
{
	if (m_texture)
		oroDestroyTextureObject(m_texture);

	if (m_texture_array)
		oroFree(m_texture_array);
}

void OrochiTexture3D::operator=(OrochiTexture3D&& other) noexcept
{
	m_texture_array = std::move(other.m_texture_array);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
	other.m_texture_array = nullptr;
}

void OrochiTexture3D::init_from_images(const std::vector<Image8Bit>& images, HIPfilter_mode filtering_mode)
{
	int channels = images[0].channels;
	if (channels == 1 || channels == 3)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "1-channel & 3-channels textures not supported on the GPU yet.");

		return;
	}

	width = images[0].width;
	height = images[0].height;
	depth = images.size();

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(unsigned char) * 8, sizeof(unsigned char) * 8, sizeof(unsigned char) * 8, sizeof(unsigned char) * 8, oroChannelFormatKindUnsigned);
	OROCHI_CHECK_ERROR(oroMalloc3DArray(&m_texture_array, &channel_descriptor, oroExtent{ width, height, depth }, oroArrayDefault));

	// Because we'r ecopying to a CUDA/HIP array, we need the input data
	// to be in a single linear block of data
	std::vector<float> linear_image_data(width * height * depth);
	for (int i = 0; i < images.size(); i++)
		std::copy(images[i].data().begin(), images[i].data().end(), linear_image_data.begin() + width * height * i);

	oroMemcpy3DParms copyParams = { 0 };
	copyParams.dstArray = m_texture_array;
	copyParams.extent = { width, height, depth };
	copyParams.kind = oroMemcpyHostToDevice;
	copyParams.srcPtr = oroPitchedPtr{ linear_image_data.data(), width, width, height };
	OROCHI_CHECK_ERROR(oroMemcpy3D(&copyParams));

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

void OrochiTexture3D::init_from_images(const std::vector<Image32Bit>& images, HIPfilter_mode filtering_mode)
{
	int channels = images[0].channels;
	if (channels == 1 || channels == 3)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "1-channel & 3-channels textures not supported on the GPU yet.");

		return;
	}

	width = images[0].width;
	height = images[0].height;
	depth = images.size();

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, oroChannelFormatKindUnsigned);
	OROCHI_CHECK_ERROR(oroMalloc3DArray(&m_texture_array, &channel_descriptor, oroExtent{ width, height, depth }, oroArrayDefault));

	// Because we'r ecopying to a CUDA/HIP array, we need the input data
	// to be in a single linear block of data
	std::vector<float> linear_image_data(width * height * depth * channels);
	for (int i = 0; i < images.size(); i++)
		std::copy(images[i].data().begin(), images[i].data().end(), linear_image_data.begin() + width * height * i * 4);

	oroMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = oroPitchedPtr{ linear_image_data.data(), width * 4 * sizeof(float), width * 4, height};
	copyParams.dstArray = m_texture_array;
	copyParams.extent = { width, height, depth };
	copyParams.kind = oroMemcpyHostToDevice;
	OROCHI_CHECK_ERROR(oroMemcpy3D(&copyParams));

	// Resource descriptor
	ORO_RESOURCE_DESC resource_descriptor;
	std::memset(&resource_descriptor, 0, sizeof(resource_descriptor));
	resource_descriptor.resType = ORO_RESOURCE_TYPE_ARRAY;
	resource_descriptor.res.array.hArray = m_texture_array;

	ORO_TEXTURE_DESC texture_descriptor;
	std::memset(&texture_descriptor, 0, sizeof(texture_descriptor));
	texture_descriptor.addressMode[0] = ORO_TR_ADDRESS_MODE_CLAMP;
	texture_descriptor.addressMode[1] = ORO_TR_ADDRESS_MODE_CLAMP;
	texture_descriptor.addressMode[2] = ORO_TR_ADDRESS_MODE_CLAMP;
	texture_descriptor.filterMode = filtering_mode;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
}

oroTextureObject_t OrochiTexture3D::get_device_texture()
{
	return m_texture;
}
