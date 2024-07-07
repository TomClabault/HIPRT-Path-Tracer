/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMAGE_H
#define IMAGE_H

#include "HostDeviceCommon/Color.h"
#include "Utils/Utils.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <string>
#include <type_traits>

struct ImageBin
{
    int x0, x1;
    int y0, y1;
};

template <typename Derived, typename PixelType, int NbComponents>
class ImageBase
{
public:
    ImageBase() : width(0), height(0) {}
    ImageBase(int width, int height);
    ImageBase(PixelType* data, int width, int height);
    ImageBase(const std::vector<PixelType>& data, int width, int height);
    ImageBase(const std::string& filepath) : ImageBase(filepath.c_str()) {}
    ImageBase(const char* filepath) { *this = ImageBase::read_image(filepath, false); };
    
    static Derived read_image(const std::string& filepath, bool flipY);
    static Derived read_image_hdr(const std::string& filepath, bool flipY);

    bool write_image_png(const char* filename, const bool flipY = true) const;
    bool write_image_hdr(const char* filename, const bool flipY = true) const;

    float luminance_of_pixel(int x, int y) const;
    float luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const;
    float luminance_of_area(const ImageBin& area) const;

    ColorRGBA32F sample_rgba32f(float2 uv) const;

    void compute_cdf();
    std::vector<float> compute_get_cdf();
    const std::vector<float>& get_cdf() const;
    std::vector<float>& get_cdf();

    size_t byte_size() const;

    void set_data(const std::vector<PixelType>& data);
    const std::vector<PixelType>& data() const;
    std::vector<PixelType>& data();

    const PixelType& operator[](int index) const;
    PixelType& operator[](int index);

    int width = 0, height = 0;

protected:
    std::vector<PixelType> m_pixel_data;
    std::vector<float> m_cdf;

    bool m_cdf_computed = false;
};

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
ImageBase<Derived, PixelType, NbComponents>::ImageBase(int width, int height) : ImageBase(std::vector<PixelType>(width * height * NbComponents, 0), width, height) {}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
ImageBase<Derived, PixelType, NbComponents>::ImageBase(PixelType* data, int width, int height) : width(width), height(height)
{
    m_pixel_data = std::vector<PixelType>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height]);
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
ImageBase<Derived, PixelType, NbComponents>::ImageBase(const std::vector<PixelType>& data, int width, int height) : width(width), height(height), m_pixel_data(data) {}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
Derived ImageBase<Derived, PixelType, NbComponents>::read_image(const std::string& filepath, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &channels, NbComponents);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    Derived output_image(width, height);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            float divider = 0.0f;
            if (std::is_same<PixelType, float>())
                divider = 255.0f;
            else if (std::is_same<PixelType, unsigned char>())
                divider = 1.0f;

            for (int i = 0; i < NbComponents; i++)
                output_image[index * NbComponents + i] = pixels[index * NbComponents + i] / divider;
        }
    }

    return output_image;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
Derived ImageBase<Derived, PixelType, NbComponents>::read_image_hdr(const std::string& filepath, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &channels, NbComponents);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    std::vector<PixelType> converted_data(width * height * NbComponents);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            for (int i = 0; i < NbComponents; i++)
            {
                converted_data[index * NbComponents + i] = static_cast<PixelType>(pixels[index * NbComponents + i]);
            }
        }
    }

    return Derived(converted_data, width, height);
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
bool ImageBase<Derived, PixelType, NbComponents>::write_image_png(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    float multiplier = 0.0f;
    if (std::is_same<PixelType, float>())
        multiplier = 255.0f;
    else if (std::is_same<PixelType, unsigned char>())
        multiplier = 1.0f;

    std::vector<unsigned char> tmp(width * height * NbComponents);
    for (unsigned i = 0; i < width * height; i++)
    {
        for (int j = 0; j < NbComponents; j++)
        {
            tmp[i * NbComponents + j] = hippt::clamp(0.0f, 255.0f, m_pixel_data[i * NbComponents + j] * multiplier);
        }
    }

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, width, height, NbComponents, tmp.data(), width * NbComponents) != 0;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
bool ImageBase<Derived, PixelType, NbComponents>::write_image_hdr(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    float divider = 0.0f;
    if (std::is_same<PixelType, float>)
        divider = 1.0f;
    else if (std::is_same<PixelType, unsigned char>)
        divider = 255.0f;

    std::vector<float> tmp(width * height * NbComponents);
    for (unsigned i = 0; i < width * height; i++)
        for (int j = 0; j < NbComponents; j++)
            tmp[i * NbComponents + j] = m_pixel_data[i * NbComponents + j] / divider;

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, width, height, NbComponents, reinterpret_cast<const float*>(m_pixel_data.data())) != 0;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
float ImageBase<Derived, PixelType, NbComponents>::luminance_of_pixel(int x, int y) const
{
    int start_pixel = (x + y * width) * NbComponents;

    // Computing the luminance with a *maximum* of 3 components.
    // 
    // If the texture only has one component (i.e. only red), the following
    // loop will only loop through the red component with the right weight. 
    // 
    // If the image has more than 1 components, 3 for example, then we'll loop through
    // the 3 components and apply the weights.
    // 
    // If the image has 4 components, we will still only take RGB into account for the
    // luminance compoutation but not alpha
    float luminance = 0.0;
    float weights[3] = { 0.3086f, 0.6094f, 0.0820f };
    for (int i = 0; i < hippt::min(NbComponents, 3); i++)
        luminance += m_pixel_data[start_pixel + i] * weights[i];

    return luminance;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
float ImageBase<Derived, PixelType, NbComponents>::luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const
{
    float luminance = 0.0f;

    for (int x = start_x; x < stop_x; x++)
        for (int y = start_y; y < stop_y; y++)
            luminance += luminance_of_pixel(x, y);

    return luminance;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
float ImageBase<Derived, PixelType, NbComponents>::luminance_of_area(const ImageBin& area) const
{
    return luminance_of_area(area.x0, area.y0, area.x1, area.y1);
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
ColorRGBA32F ImageBase<Derived, PixelType, NbComponents>::sample_rgba32f(float2 uv) const
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

    ColorRGBA32F out_color;
    for (int i = 0; i < NbComponents; i++)
        out_color[i] = m_pixel_data[(x + y * width) * NbComponents + i];

    return out_color;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
void ImageBase<Derived, PixelType, NbComponents>::compute_cdf()
{
    m_cdf.resize(height * width);
    m_cdf[0] = 0.0f;

    float max_radiance = 0.0f;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            m_cdf[index] = m_cdf[std::max(index - 1, 0)] + luminance_of_pixel(x, y);

            for (int i = 0; i < hippt::min(3, NbComponents); i++)
                max_radiance = hippt::max(max_radiance, m_pixel_data[(x + y * width) * NbComponents + i]);
        }
    }

    std::cout << "Max radiance of envmap: " << max_radiance << std::endl;

    m_cdf_computed = true;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
std::vector<float> ImageBase<Derived, PixelType, NbComponents>::compute_get_cdf()
{
    compute_cdf();
    return m_cdf;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
const std::vector<float>& ImageBase<Derived, PixelType, NbComponents>::get_cdf() const
{
    if (!m_cdf_computed)
        compute_cdf();

    return m_cdf;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
std::vector<float>& ImageBase<Derived, PixelType, NbComponents>::get_cdf()
{
    if (!m_cdf_computed)
        compute_cdf();

    return m_cdf;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
size_t ImageBase<Derived, PixelType, NbComponents>::byte_size() const
{
    return width * height * sizeof(PixelType);
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
void ImageBase<Derived, PixelType, NbComponents>::set_data(const std::vector<PixelType>& data)
{
    m_pixel_data = data;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
const std::vector<PixelType>& ImageBase<Derived, PixelType, NbComponents>::data() const
{
    return m_pixel_data;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
std::vector<PixelType>& ImageBase<Derived, PixelType, NbComponents>::data()
{
    return m_pixel_data;
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
const PixelType& ImageBase<Derived, PixelType, NbComponents>::operator[](int index) const
{
    return m_pixel_data[index];
}

// -----------------------------------------------------------------
template <typename Derived, typename PixelType, int NbComponents>
PixelType& ImageBase<Derived, PixelType, NbComponents>::operator[](int index)
{
    return m_pixel_data[index];
}

class ImageRGB32F : public ImageBase<ImageRGB32F, float, 3>
{
public:
    // Inheriting constructors from ImageBase
    using ImageBase::ImageBase;

    ColorRGB32F* get_data_as_ColorRGB32F();
    ColorRGB32F get_pixel_ColorRGB32F(int pixel_index) const;
};

class ImageRGBA32F : public ImageBase<ImageRGBA32F, float, 4>
{
public:
    // Inheriting constructors from ImageBase
    using ImageBase::ImageBase;

    ColorRGBA32F* get_data_as_ColorRGBA32F();
    ColorRGBA32F get_pixel_ColorRGBA32F(int pixel_index) const;
};

#endif
