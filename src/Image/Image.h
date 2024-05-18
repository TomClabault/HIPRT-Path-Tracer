/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMAGE_H
#define IMAGE_H

#include "HostDeviceCommon/Color.h"

#include <string>

struct ImageBin
{
    int x0, x1;
    int y0, y1;
};

template <typename PixelType>
class ImageBase
{
public:
    ImageBase() : width(0), height(0) {}
    ImageBase(int width, int height) : width(width), height(height), m_pixel_data(width* height) {}
    ImageBase(PixelType* data, int width, int height);
    ImageBase(const std::vector<PixelType>& data, int width, int height) : width(width), height(height), m_pixel_data(data) {}
    
    float luminance_of_pixel(int x, int y) const;
    float luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const;
    float luminance_of_area(const ImageBin& area) const;

    PixelType sample(float2 uv) const;

    void compute_cdf();
    std::vector<float> compute_get_cdf() const;
    const std::vector<float>& get_cdf() const;
    std::vector<float>& get_cdf();

    size_t byte_size() const;

    void set_data(const std::vector<PixelType>& data);
    const std::vector<PixelType>& data() const;
    std::vector<PixelType>& data();

    const PixelType& operator[](int index) const;
    PixelType& operator[](int index);

    int width = 0, height = 0, channels = 0;

protected:
    std::vector<PixelType> m_pixel_data;
    std::vector<float> m_cdf;

    bool m_cdf_computed = false;
};

template <typename PixelType>
ImageBase<PixelType>::ImageBase(PixelType* data, int width, int height) : width(width), height(height)
{
    m_pixel_data = std::vector<PixelType>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height]);
}


template <typename PixelType>
float ImageBase<PixelType>::luminance_of_pixel(int x, int y) const
{
    PixelType pixel = m_pixel_data[y * width + x];

    return 0.3086f * pixel.r + 0.6094f * pixel.g + 0.0820f * pixel.b;
}

template <typename PixelType>
float ImageBase<PixelType>::luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const
{
    float luminance = 0.0f;

    for (int x = start_x; x < stop_x; x++)
        for (int y = start_y; y < stop_y; y++)
            luminance += luminance_of_pixel(x, y);

    return luminance;
}

template <typename PixelType>
float ImageBase<PixelType>::luminance_of_area(const ImageBin& area) const
{
    return luminance_of_area(area.x0, area.y0, area.x1, area.y1);
}

template <typename PixelType>
PixelType ImageBase<PixelType>::sample(float2 uv) const
{
    // Sampling in repeat mode so we're just keeping the fractional part
    float u = uv.x - (int)uv.x;
    float v = uv.y - (int)uv.y;

    // For negative UVs, we also want to repeat and we want, for example, 
    // -0.1f to behave as 0.9f
    u = u < 0 ? 1.0f + u : u;
    v = v < 0 ? 1.0f + v : v;

    // Sampling with [0, 0] bottom-left convention
    v = 1.0f - v;

    int x = (u * (width - 1));
    int y = (v * (height - 1));

    return m_pixel_data[x + y * width];
}

template <typename PixelType>
void ImageBase<PixelType>::compute_cdf()
{
    m_cdf.resize(height * width);
    m_cdf[0] = 0.0f;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            m_cdf[index] = m_cdf[std::max(index - 1, 0)] + luminance_of_pixel(x, y);
        }
    }

    m_cdf_computed = true;
}

template <typename PixelType>
std::vector<float> ImageBase<PixelType>::compute_get_cdf() const
{
    std::vector<float> cdf(height * width);
    cdf[0] = 0.0f;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            cdf[index] = cdf[std::max(index - 1, 0)] + luminance_of_pixel(x, y);
        }
    }

    return cdf;
}

template <typename PixelType>
const std::vector<float>& ImageBase<PixelType>::get_cdf() const
{
    if (!m_cdf_computed)
        compute_cdf();

    return m_cdf;
}

template <typename PixelType>
std::vector<float>& ImageBase<PixelType>::get_cdf()
{
    if (!m_cdf_computed)
        compute_cdf();

    return m_cdf;
}

template <typename PixelType>
size_t ImageBase<PixelType>::byte_size() const
{
    return width * height * sizeof(PixelType);
}

template <typename PixelType>
void ImageBase<PixelType>::set_data(const std::vector<PixelType>& data)
{
    m_pixel_data = data;
}

template <typename PixelType>
const std::vector<PixelType>& ImageBase<PixelType>::data() const
{
    return m_pixel_data;
}

template <typename PixelType>
std::vector<PixelType>& ImageBase<PixelType>::data()
{
    return m_pixel_data;
}

template <typename PixelType>
const PixelType& ImageBase<PixelType>::operator[](int index) const
{
    return m_pixel_data[index];
}

template <typename PixelType>
PixelType& ImageBase<PixelType>::operator[](int index)
{
    return m_pixel_data[index];
}

class Image : public ImageBase<ColorRGB>
{
public:
    Image() : ImageBase() {}
    Image(int width, int height) : ImageBase(width, height) { channels = 3; }
    Image(ColorRGB* data, int width, int height) : ImageBase(data, width, height) { channels = 3; }
    Image(const std::vector<ColorRGB>& data, int width, int height) : ImageBase(data, width, height) { channels = 3; }
    Image(const std::string& filepath);
    Image(const char* filepath);

    static Image read_image(const std::string& filepath, bool flipY);
    static Image read_image_hdr(const std::string& filepath, bool flipY);
    bool write_image_png(const char* filename, const bool flipY = true) const;
    bool write_image_hdr(const char* filename, const bool flipY = true) const;
};

class ImageRGBA : public ImageBase<ColorRGBA>
{
public:
    ImageRGBA() : ImageBase() {}
    ImageRGBA(int width, int height) : ImageBase(width, height) { channels = 4; }
    ImageRGBA(ColorRGBA* data, int width, int height) : ImageBase(data, width, height) { channels = 4; }
    ImageRGBA(const std::vector<ColorRGBA>& data, int width, int height) : ImageBase(data, width, height) { channels = 4; }
    ImageRGBA(const std::string& filepath);
    ImageRGBA(const char* filepath);

    static ImageRGBA read_image(const std::string& filepath, bool flipY);
    static ImageRGBA read_image_hdr(const std::string& filepath, bool flipY);
    bool write_image_png(const char* filename, const bool flipY = true) const;
    bool write_image_hdr(const char* filename, const bool flipY = true) const;
};

#endif
