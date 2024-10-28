/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_TEXTURE_3D_H
#define OROCHI_TEXTURE_3D_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Image/Image.h"

#include <vector>

class OrochiTexture3D
{
public:
	OrochiTexture3D() {}
	OrochiTexture3D(const std::vector<Image8Bit>& image, HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_POINT);
	OrochiTexture3D(const std::vector<Image32Bit>& image, HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_POINT);
	OrochiTexture3D(const OrochiTexture3D& other) = delete;
	OrochiTexture3D(OrochiTexture3D&& other) noexcept;
	~OrochiTexture3D();

	void operator=(const OrochiTexture3D& other) = delete;
	void operator=(OrochiTexture3D&& other) noexcept;

	void init_from_images(const std::vector<Image8Bit>& images, HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_POINT);
	void init_from_images(const std::vector<Image32Bit>& images, HIPfilter_mode filtering_mode = ORO_TR_FILTER_MODE_POINT);

	oroTextureObject_t get_device_texture();

	unsigned int width = 0, height = 0, depth = 0;

private:
	oroArray_t m_texture_array = nullptr;

	oroTextureObject_t m_texture = nullptr;
};

#endif
