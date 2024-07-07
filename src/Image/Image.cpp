/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"

// This CPP file is used to define the STBI implementation once and for all
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

ColorRGB32F* ImageRGB32F::get_data_as_ColorRGB32F()
{
    return reinterpret_cast<ColorRGB32F*>(m_pixel_data.data());
}

ColorRGB32F ImageRGB32F::get_pixel_ColorRGB32F(int pixel_index) const
{
    return ColorRGB32F(m_pixel_data[pixel_index * 3 + 0], m_pixel_data[pixel_index * 3 + 1], m_pixel_data[pixel_index * 3 + 2]);
}

ColorRGBA32F* ImageRGBA32F::get_data_as_ColorRGBA32F()
{
    return reinterpret_cast<ColorRGBA32F*>(m_pixel_data.data());
}

ColorRGBA32F ImageRGBA32F::get_pixel_ColorRGBA32F(int pixel_index) const
{
    return ColorRGBA32F(m_pixel_data[pixel_index * 4 + 0], m_pixel_data[pixel_index * 4 + 1], m_pixel_data[pixel_index * 4 + 2], m_pixel_data[pixel_index * 4 + 3]);
}
