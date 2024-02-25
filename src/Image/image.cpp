#include "Kernels/includes/hiprt_color.h"
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


inline float clamp(const float x, const float min, const float max)
{
    if (x < min) return min;
    else if (x > max) return max;
    else return x;
}

Image::Image(HIPRTColor* data, int width, int height) : width(width), height(height)
{
    m_pixel_data = std::vector<HIPRTColor>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height]);
}

size_t Image::byte_size() const
{
    return width * height * sizeof(HIPRTColor);
}

float Image::luminance_of_pixel(int x, int y) const
{
    HIPRTColor pixel = m_pixel_data[y * width + x];

    return 0.3086 * pixel.r + 0.6094 * pixel.g + 0.0820 * pixel.b;
}

float Image::luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const
{
    float luminance = 0.0f;

    for (int x = start_x; x < stop_x; x++)
        for (int y = start_y; y < stop_y; y++)
            luminance += luminance_of_pixel(x, y);

    return luminance;
}

float Image::luminance_of_area(const ImageBin& area) const
{
    return luminance_of_area(area.x0, area.y0, area.x1, area.y1);
}

bool Image::write_image_png(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<unsigned char> tmp(byte_size());
    for (unsigned i = 0; i < width * height; i++)
    {
        HIPRTColor pixel = m_pixel_data[i] * 255;

        tmp[i * 3 + 0] = clamp(pixel.r, 0, 255);
        tmp[i * 3 + 1] = clamp(pixel.g, 0, 255);
        tmp[i * 3 + 2] = clamp(pixel.b, 0, 255);
    }

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, width, height, 3, tmp.data(), width * 3) != 0;
}

bool Image::write_image_hdr(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, width, height, 3, reinterpret_cast<const float*>(m_pixel_data.data())) != 0;
}

void Image::set_data(const std::vector<HIPRTColor>& data)
{
    m_pixel_data = data;
}

const std::vector<HIPRTColor>& Image::data() const
{
    return m_pixel_data;
}

std::vector<HIPRTColor>& Image::data()
{
    return m_pixel_data;
}

const HIPRTColor& Image::operator[](int index) const
{
    return m_pixel_data[index];
}

HIPRTColor& Image::operator[](int index)
{
    return m_pixel_data[index];
}
