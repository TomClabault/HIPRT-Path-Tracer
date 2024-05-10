#include "HIPRT-Orochi/OrochiTexture.h"

#include <Orochi/Orochi.h>

OrochiTexture::OrochiTexture(const ImageRGBA& image)
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

void OrochiTexture::init_from_image(const ImageRGBA& image)
{
	width = image.width;
	height = image.height;

	oroChannelFormatDesc channelDescriptor = oroCreateChannelDesc(sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, oroChannelFormatKindFloat);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channelDescriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * image.channels * sizeof(float), image.width * sizeof(float) * image.channels, image.height, oroMemcpyHostToDevice));

	// Resource descriptor
	oroResourceDesc resource_descriptor;
	std::memset(&resource_descriptor, 0, sizeof(resource_descriptor));
	resource_descriptor.resType = hipResourceTypeArray;
	resource_descriptor.res.array.array = m_texture_array;

	oroTextureDesc texture_descriptor;
	std::memset(&texture_descriptor, 0, sizeof(texture_descriptor));
	texture_descriptor.addressMode[0] = hipAddressModeWrap;
	texture_descriptor.addressMode[1] = hipAddressModeWrap;
	texture_descriptor.filterMode = hipFilterModePoint;
	texture_descriptor.normalizedCoords = 1;

	OROCHI_CHECK_ERROR(oroCreateTextureObject(&m_texture, &resource_descriptor, &texture_descriptor, nullptr));
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}
