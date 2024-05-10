#include "HIPRT-Orochi/OrochiTexture.h"

#include <Orochi/Orochi.h>

OrochiTexture::OrochiTexture(const ImageRGBA& image)
{
	width = image.width;
	height = image.height;

	m_image_buffer.resize(image.width * image.height * image.channels);
	m_image_buffer.upload_data((void*)image.data().data());

	oroChannelFormatDesc channelDescriptor = oroCreateChannelDesc(sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, sizeof(float) * 8, oroChannelFormatKindFloat);
	OROCHI_CHECK_ERROR(oroMallocArray(&m_texture_array, &channelDescriptor, image.width, image.height, oroArrayDefault));
	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(m_texture_array, 0, 0, image.data().data(), image.width * image.channels * sizeof(float), image.width * sizeof(float) * image.channels, image.height, oroMemcpyHostToDevice));

	// Resource descriptor
	ORO_RESOURCE_DESC resDesc;
	std::memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = ORO_RESOURCE_TYPE_ARRAY;
	resDesc.res.array.hArray = m_texture_array;
	/*resDesc.res.pitch2D.devPtr = reinterpret_cast<oroDeviceptr_t>(m_image_buffer.get_device_pointer());
	resDesc.res.pitch2D.format = ORO_AD_FORMAT_FLOAT;
	resDesc.res.pitch2D.numChannels = image.channels;
	resDesc.res.pitch2D.width = image.width;
	resDesc.res.pitch2D.height = image.height;
	resDesc.res.pitch2D.pitchInBytes = image.width * sizeof(float) * image.channels;*/

	ORO_TEXTURE_DESC texDesc;
	std::memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = ORO_TR_ADDRESS_MODE_WRAP;
	texDesc.addressMode[1] = ORO_TR_ADDRESS_MODE_WRAP;
	texDesc.filterMode = ORO_TR_FILTER_MODE_POINT;
	texDesc.flags = ORO_TRSF_NORMALIZED_COORDINATES;

	OROCHI_CHECK_ERROR(oroTexObjectCreate(&m_texture, &resDesc, &texDesc, nullptr));
}

OrochiTexture::OrochiTexture(OrochiTexture&& other)
{
	m_image_buffer = std::move(other.m_image_buffer);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
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
	m_image_buffer = std::move(other.m_image_buffer);
	m_texture = std::move(other.m_texture);

	other.m_texture = nullptr;
}

oroTextureObject_t OrochiTexture::get_device_texture()
{
	return m_texture;
}
