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
	ORO_RESOURCE_DESC resDesc;
	std::memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = ORO_RESOURCE_TYPE_ARRAY;
	resDesc.res.array.hArray = m_texture_array;

	ORO_TEXTURE_DESC texDesc;
	std::memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = ORO_TR_ADDRESS_MODE_WRAP;
	texDesc.addressMode[1] = ORO_TR_ADDRESS_MODE_WRAP;
	texDesc.filterMode = ORO_TR_FILTER_MODE_POINT;
	texDesc.flags = ORO_TRSF_NORMALIZED_COORDINATES;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resDesc, &texDesc, nullptr));
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}
