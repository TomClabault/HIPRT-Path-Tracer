#ifndef OROCHI_TEXTURE_H
#define OROCHI_TEXTURE_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Image/Image.h"

class OrochiTexture
{
public:
	OrochiTexture() {}
	OrochiTexture(const ImageRGBA& image);
	OrochiTexture(const OrochiTexture& other) = delete;
	OrochiTexture(OrochiTexture&& other);
	~OrochiTexture();

	void operator=(const OrochiTexture& other) = delete;
	void operator=(OrochiTexture&& other);

	void init_from_image(const ImageRGBA& image);
	oroTextureObject_t get_device_texture();

	unsigned int width = 0, height = 0;

private:
	oroArray_t m_texture_array = nullptr;

	oroTextureObject_t m_texture = nullptr;
};

#endif
