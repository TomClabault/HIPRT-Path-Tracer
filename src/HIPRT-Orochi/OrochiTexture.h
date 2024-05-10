#ifndef OROCHI_TEXTURE_H
#define OROCHI_TEXTURE_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Image/Image.h"

class OrochiTexture
{
public:
	OrochiTexture() {}
	OrochiTexture(const ImageRGBA& image);
	OrochiTexture(OrochiTexture&& other);
	~OrochiTexture();

	void operator=(OrochiTexture&& other);

	oroTextureObject_t get_device_texture();

	unsigned int width = 0, height = 0;

private:
	OrochiBuffer<float> m_image_buffer;
	oroArray_t m_texture_array;

	oroTextureObject_t m_texture = nullptr;
};

#endif
