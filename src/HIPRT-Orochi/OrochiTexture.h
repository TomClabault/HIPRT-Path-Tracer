/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_TEXTURE_H
#define OROCHI_TEXTURE_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Image/Image.h"

class OrochiTexture
{
public:
	OrochiTexture() {}
	OrochiTexture(const ImageRGBA32F& image);
	OrochiTexture(const OrochiTexture& other) = delete;
	OrochiTexture(OrochiTexture&& other);
	~OrochiTexture();

	void operator=(const OrochiTexture& other) = delete;
	void operator=(OrochiTexture&& other);

	template <typename Derived, typename PixelType, int NbComponents>
	void init_from_image(const ImageBase<Derived, PixelType, NbComponents>& image);

	oroTextureObject_t get_device_texture();
	oroTextureObject_t* get_device_texture_pointer();

	unsigned int width = 0, height = 0;

private:
	oroArray_t m_texture_array = nullptr;

	oroTextureObject_t m_texture = nullptr;
};

template <typename Derived, typename PixelType, int NbComponents>
void OrochiTexture::init_from_image(const ImageBase<Derived, PixelType, NbComponents>& image)
{
	width = image.width;
	height = image.height;

	hipChannelFormatKind formatKind;
	if (std::is_same<PixelType, float>())
		formatKind = oroChannelFormatKindFloat;
	else if (std::is_same<PixelType, unsigned char>())
		formatKind = oroChannelFormatKindUnsigned;

	// X, Y, Z and W in oroCreateChannelDesc are the number of *bits* of each component
	oroChannelFormatDesc channel_descriptor = oroCreateChannelDesc(sizeof(PixelType) * 8, sizeof(PixelType) * 8, sizeof(PixelType) * 8, sizeof(PixelType) * 8, formatKind);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channel_descriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * NbComponents * sizeof(PixelType), image.width * sizeof(PixelType) * NbComponents, image.height, oroMemcpyHostToDevice));

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

#endif
