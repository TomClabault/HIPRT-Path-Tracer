/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Utils/Utils.h"

// This CPP file is used to define the STBI implementation once and for all
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

Image8Bit::Image8Bit(int width, int height, int channels) : Image8Bit(std::vector<unsigned char>(width * height * channels, 0), width, height, channels) {}

Image8Bit::Image8Bit(unsigned char* data, int width, int height, int channels) : width(width), height(height), channels(channels)
{
    m_pixel_data = std::vector<unsigned char>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height]);
}

Image8Bit::Image8Bit(const std::vector<unsigned char>& data, int width, int height, int channels) : width(width), height(height), channels(channels), m_pixel_data(data) {}

Image8Bit Image8Bit::read_image(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, read_channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    Image8Bit output_image(width, height, output_channels);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            for (int i = 0; i < output_channels; i++)
                output_image[index * output_channels + i] = pixels[index * output_channels + i];
        }
    }

    return output_image;
}

Image8Bit Image8Bit::read_image_hdr(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, read_channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    std::vector<unsigned char> converted_data(width * height * output_channels);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            for (int i = 0; i < output_channels; i++)
            {
                converted_data[index * output_channels + i] = static_cast<unsigned char>(pixels[index * output_channels + i]);
            }
        }
    }

    return Image8Bit(converted_data, width, height, output_channels);
}

bool Image8Bit::write_image_png(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<unsigned char> tmp(width * height * channels);
    for (unsigned i = 0; i < width * height; i++)
        for (int j = 0; j < channels; j++)
            tmp[i * channels + j] = hippt::clamp(static_cast<unsigned char>(0), static_cast<unsigned char>(255), m_pixel_data[i * channels + j]);

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, width, height, channels, tmp.data(), width * channels) != 0;
}

bool Image8Bit::write_image_hdr(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<float> tmp(width * height * channels);
    for (unsigned i = 0; i < width * height; i++)
        for (int j = 0; j < channels; j++)
            tmp[i * channels + j] = m_pixel_data[i * channels + j] / 255.0f;

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, width, height, channels, reinterpret_cast<const float*>(m_pixel_data.data())) != 0;
}

float Image8Bit::luminance_of_pixel(int x, int y) const
{
    int start_pixel = (x + y * width) * channels;

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
    for (int i = 0; i < hippt::min(channels, 3); i++)
        luminance += m_pixel_data[start_pixel + i] * weights[i];

    return luminance;
}

float Image8Bit::luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const
{
    float luminance = 0.0f;

    for (int x = start_x; x < stop_x; x++)
        for (int y = start_y; y < stop_y; y++)
            luminance += luminance_of_pixel(x, y);

    return luminance;
}

float Image8Bit::luminance_of_area(const ImageBin& area) const
{
    return luminance_of_area(area.x0, area.y0, area.x1, area.y1);
}

ColorRGBA32F Image8Bit::sample_rgba32f(float2 uv) const
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
    for (int i = 0; i < channels; i++)
        out_color[i] = m_pixel_data[(x + y * width) * channels + i] / 255.0f;

    return out_color;
}

void Image8Bit::set_data(const std::vector<unsigned char>& data)
{
    m_pixel_data = data;
}

const std::vector<unsigned char>& Image8Bit::data() const
{
    return m_pixel_data;
}

std::vector<unsigned char>& Image8Bit::data()
{
    return m_pixel_data;
}

const unsigned char& Image8Bit::operator[](int index) const
{
    return m_pixel_data[index];
}

unsigned char& Image8Bit::operator[](int index)
{
    return m_pixel_data[index];
}

void Image8Bit::compute_cdf()
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

            for (int i = 0; i < hippt::min(3, channels); i++)
                max_radiance = hippt::max(max_radiance, static_cast<float>(m_pixel_data[(x + y * width) * channels + i]));
        }
    }

    std::cout << "Max radiance of envmap: " << max_radiance << std::endl;

    m_cdf_computed = true;
}

std::vector<float> Image8Bit::compute_get_cdf()
{
    compute_cdf();
    return m_cdf;
}

std::vector<float>& Image8Bit::get_cdf()
{
    if (!m_cdf_computed)
        compute_cdf();

    return m_cdf;
}

size_t Image8Bit::byte_size() const
{
    return width * height * sizeof(unsigned char);
}























Image32Bit::Image32Bit(int width, int height, int channels) : Image32Bit(std::vector<float>(width* height* channels, 0), width, height, channels) {}

Image32Bit::Image32Bit(float* data, int width, int height, int channels) : width(width), height(height), channels(channels)
{
    m_pixel_data = std::vector<float>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height]);
}

Image32Bit::Image32Bit(const std::vector<float>& data, int width, int height, int channels) : width(width), height(height), channels(channels), m_pixel_data(data) {}

Image32Bit Image32Bit::read_image(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, read_channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    Image32Bit output_image(width, height, output_channels);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            for (int i = 0; i < output_channels; i++)
                output_image[index * output_channels + i] = pixels[index * output_channels + i] / 255.0f;
        }
    }

    return output_image;
}

Image32Bit Image32Bit::read_image_hdr(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, read_channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    std::vector<float> converted_data(width * height * output_channels);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;

            for (int i = 0; i < output_channels; i++)
                converted_data[index * output_channels + i] = pixels[index * output_channels + i];
        }
    }

    return Image32Bit(converted_data, width, height, output_channels);
}

bool Image32Bit::write_image_png(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<unsigned char> tmp(width * height * channels);
    for (unsigned i = 0; i < width * height * channels; i++)
        tmp[i] = hippt::clamp(0.0f, 255.0f, m_pixel_data[i] * 255.0f);

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, width, height, channels, tmp.data(), width * channels) != 0;
}

bool Image32Bit::write_image_hdr(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<float> tmp(width * height * channels);
    for (unsigned i = 0; i < width * height; i++)
        for (int j = 0; j < channels; j++)
            tmp[i * channels + j] = m_pixel_data[i * channels + j];

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, width, height, channels, reinterpret_cast<const float*>(m_pixel_data.data())) != 0;
}

float Image32Bit::luminance_of_pixel(int x, int y) const
{
    int start_pixel = (x + y * width) * channels;

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
    for (int i = 0; i < hippt::min(channels, 3); i++)
        luminance += m_pixel_data[start_pixel + i] * weights[i];

    return luminance;
}

float Image32Bit::luminance_of_area(int start_x, int start_y, int stop_x, int stop_y) const
{
    float luminance = 0.0f;

    for (int x = start_x; x < stop_x; x++)
        for (int y = start_y; y < stop_y; y++)
            luminance += luminance_of_pixel(x, y);

    return luminance;
}

float Image32Bit::luminance_of_area(const ImageBin& area) const
{
    return luminance_of_area(area.x0, area.y0, area.x1, area.y1);
}

ColorRGBA32F Image32Bit::sample_rgba32f(float2 uv) const
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
    for (int i = 0; i < channels; i++)
        out_color[i] = m_pixel_data[(x + y * width) * channels + i];

    return out_color;
}

void Image32Bit::set_data(const std::vector<float>& data)
{
    m_pixel_data = data;
}

const std::vector<float>& Image32Bit::data() const
{
    return m_pixel_data;
}

std::vector<float>& Image32Bit::data()
{
    return m_pixel_data;
}

const float& Image32Bit::operator[](int index) const
{
    return m_pixel_data[index];
}

float& Image32Bit::operator[](int index)
{
    return m_pixel_data[index];
}

void Image32Bit::compute_cdf()
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

            for (int i = 0; i < hippt::min(3, channels); i++)
                max_radiance = hippt::max(max_radiance, m_pixel_data[(x + y * width) * channels + i]);
        }
    }

    std::cout << "Max radiance of envmap: " << max_radiance << std::endl;

    m_cdf_computed = true;
}

std::vector<float> Image32Bit::compute_get_cdf()
{
    compute_cdf();
    return m_cdf;
}

std::vector<float>& Image32Bit::get_cdf()
{
    if (!m_cdf_computed)
        compute_cdf();

    return m_cdf;
}

size_t Image32Bit::byte_size() const
{
    return width * height * sizeof(unsigned char);
}

ColorRGB32F* Image32Bit::get_data_as_ColorRGB32F()
{
    return reinterpret_cast<ColorRGB32F*>(m_pixel_data.data());
}

ColorRGB32F Image32Bit::get_pixel_ColorRGB32F(int pixel_index) const
{
    return ColorRGB32F(m_pixel_data[pixel_index * 3 + 0], m_pixel_data[pixel_index * 3 + 1], m_pixel_data[pixel_index * 3 + 2]);
}

ColorRGBA32F* Image32Bit::get_data_as_ColorRGBA32F()
{
    return reinterpret_cast<ColorRGBA32F*>(m_pixel_data.data());
}

ColorRGBA32F Image32Bit::get_pixel_ColorRGBA32F(int pixel_index) const
{
    return ColorRGBA32F(m_pixel_data[pixel_index * 4 + 0], m_pixel_data[pixel_index * 4 + 1], m_pixel_data[pixel_index * 4 + 2], m_pixel_data[pixel_index * 4 + 3]);
}
