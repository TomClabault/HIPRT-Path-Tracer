/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

extern ImGuiLogger g_imgui_logger;

// This CPP file is used to define the STBI implementation once and for all
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "tinyexr.cc"

#include <deque>

Image8Bit::Image8Bit(int width, int height, int channels) : Image8Bit(std::vector<unsigned char>(width * height * channels, 0), width, height, channels) {}

Image8Bit::Image8Bit(const unsigned char* data, int width, int height, int channels) : width(width), height(height), channels(channels)
{
    m_pixel_data = std::vector<unsigned char>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height * channels]);
}

Image8Bit::Image8Bit(const std::vector<unsigned char>& data, int width, int height, int channels) : width(width), height(height), channels(channels), m_pixel_data(data) {}

Image8Bit Image8Bit::read_image(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load_thread(flipY);

    int width, height, read_channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Error reading image %s: %s", filepath.c_str(), stbi_failure_reason());
        return Image8Bit();
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

    stbi_image_free(pixels);
    return output_image;
}

Image8Bit Image8Bit::read_image_hdr(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, read_channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Error reading image %s: %s", filepath.c_str(), stbi_failure_reason());
        return Image8Bit();
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

    stbi_image_free(pixels);
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
    // luminance computation but not alpha
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
    float u = uv.x;
    if (u != 1.0f)
        // Only doing that if u != 1.0f because if we actually have
        // uv.x == 1.0f, then subtracting static_cast<int>(uv.x) will
        // give us 0.0f even though we actually want 1.0f (which is correct).
        // 
        // Basically, 1.0f gets transformed into 0.0f even though 1.0f is a correct
        // U coordinate which needs not to be wrapped
        u -= static_cast<int>(uv.x);

    float v = uv.y;
    if (v != 1.0f)
        // Same for v
        v -= static_cast<int>(uv.y);

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

std::vector<float> Image8Bit::compute_cdf() const
{
    std::vector<float> out_cdf;
    out_cdf.resize(height * width);
    out_cdf[0] = 0.0f;

    float max_radiance = 0.0f;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            out_cdf[index] = out_cdf[std::max(index - 1, 0)] + luminance_of_pixel(x, y);

            for (int i = 0; i < hippt::min(3, channels); i++)
                max_radiance = hippt::max(max_radiance, static_cast<float>(m_pixel_data[(x + y * width) * channels + i]));
        }
    }

    return out_cdf;
}

size_t Image8Bit::byte_size() const
{
    return width * height * sizeof(unsigned char);
}

bool Image8Bit::is_constant_color(int threshold) const
{
    if (width == 0 || height == 0)
        // Incorrect image
        return false;

    std::vector<unsigned char> first_pixel_color(channels);
    for (int i = 0; i < channels; i++)
        first_pixel_color[i] = m_pixel_data[i];

    std::atomic<bool> different_pixel_found = false;

    // Comparing the first pixel to all pixels of the texture and returning as soon as we find one
    // that is not within the threshold
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int i = 0; i < channels; i++)
                if (std::abs(first_pixel_color[i] - m_pixel_data[(y * width + x) * channels + i]) > threshold)
                    return false;

    return true;
}

void Image8Bit::free()
{
    m_pixel_data.clear();
    width = 0;
    height = 0;
    channels = 0;
}






















Image32Bit::Image32Bit(int width, int height, int channels) : Image32Bit(std::vector<float>(width * height * channels, 0), width, height, channels) {}

Image32Bit::Image32Bit(const float* data, int width, int height, int channels) : width(width), height(height), channels(channels)
{
    m_pixel_data = std::vector<float>();
    m_pixel_data.insert(m_pixel_data.end(), &data[0], &data[width * height * channels]);
}

Image32Bit::Image32Bit(const std::vector<float>& data, int width, int height, int channels) : width(width), height(height), channels(channels), m_pixel_data(data) {}

Image32Bit Image32Bit::read_image(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, read_channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Error reading image %s: %s", filepath.c_str(), stbi_failure_reason());
        return Image32Bit();
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

    stbi_image_free(pixels);
    return output_image;
}

Image32Bit Image32Bit::read_image_hdr(const std::string& filepath, int output_channels, bool flipY)
{
    stbi_set_flip_vertically_on_load_thread(flipY);

    int width, height, read_channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &read_channels, output_channels);

    if (!pixels)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Error reading image %s: %s", filepath.c_str(), stbi_failure_reason());
        return Image32Bit();
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

    stbi_image_free(pixels);
    return Image32Bit(converted_data, width, height, output_channels);
}

Image32Bit Image32Bit::read_image_exr(const std::string& filepath, bool flipY)
{
    float* out;
    int width;
    int height;
    const char* err = nullptr;

    int ret = LoadEXR(&out, &width, &height, filepath.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) 
    {
        if (err) 
        {
            g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Error reading EXR image: %s", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }

        return Image32Bit();
    }
    else 
    {
        if (!flipY)
        {
            std::vector<float> vector_data(out, out + width * height * 4);
            std::free(out); // release memory of image data

            return Image32Bit(vector_data, width, height, 4);
        }
        else
        {
            std::vector<float> vector_data(width * height * 4);

            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = 0; x < width; x++)
                {
                    int index_y_flipped = x + (height - 1 - y) * width;
                    int index = x + y * width;

                    index *= 4; // for RGBA
                    index_y_flipped *= 4; // for RGBA

                    vector_data[index + 0] = out[index_y_flipped + 0];
                    vector_data[index + 1] = out[index_y_flipped + 1];
                    vector_data[index + 2] = out[index_y_flipped + 2];
                    vector_data[index + 3] = out[index_y_flipped + 3];
                }
            }

            std::free(out); // release memory of image data
            return Image32Bit(vector_data, width, height, 4);
        }
    }
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
    // luminance computation but not alpha
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
    float u = uv.x;
    if (u != 1.0f)
        // Only doing that if u != 1.0f because if we actually have
        // uv.x == 1.0f, then subtracting static_cast<int>(uv.x) will
        // give us 0.0f even though we actually want 1.0f (which is correct).
        // 
        // Basically, 1.0f gets transformed into 0.0f even though 1.0f is a correct
        // U coordinate which needs not to be wrapped
        u -= static_cast<int>(uv.x);

    float v = uv.y;
    if (v != 1.0f)
        // Same for v
        v -= static_cast<int>(uv.y);

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

std::vector<float> Image32Bit::compute_cdf() const
{
    std::vector<float> out_cdf;
    out_cdf.resize(height * width);
    out_cdf[0] = 0.0f;

    float max_radiance = 0.0f;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            out_cdf[index] = out_cdf[std::max(index - 1, 0)] + luminance_of_pixel(x, y);

            for (int i = 0; i < hippt::min(3, channels); i++)
                max_radiance = hippt::max(max_radiance, m_pixel_data[(x + y * width) * channels + i]);
        }
    }

    return out_cdf;
}

/**
 * Reference: Vose's Alias Method [https://www.keithschwarz.com/darts-dice-coins/]
 */
void Image32Bit::compute_alias_table(std::vector<float>& out_probas, std::vector<int>& out_alias, float* out_luminance_total_sum) const
{
    // TODO try using floats here to reduce memory usage during the construction and see if precision is an issue or not

    // A vector of the luminance of all the pixels of the envmap
    // normalized such that the average of the elements of this vector is 'width*height'
    std::vector<double> normalized_luminance_of_pixels(width * height);
    double luminance_sum = 0.0f;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            double luminance = static_cast<double>(luminance_of_pixel(x, y));
            normalized_luminance_of_pixels[index] = luminance;
            luminance_sum += luminance;
        }
    }

    if (out_luminance_total_sum != nullptr)
        *out_luminance_total_sum = luminance_sum;

    for (double& luminance_value : normalized_luminance_of_pixels)
    {
        // Normalize so that the sum of the elements is 1
        luminance_value /= luminance_sum;

        // Scale for alias table construction such that the average of
        // the elements is 1
        luminance_value *= (width * height);
    }

    out_probas.resize(width * height);
    out_alias.resize(width * height);

    std::deque<int> small;
    std::deque<int> large;

    for (int i = 0; i < normalized_luminance_of_pixels.size(); i++)
    {
        if (normalized_luminance_of_pixels[i] < 1.0)
            small.push_back(i);
        else
            large.push_back(i);
    }

    while (!small.empty() && !large.empty())
    {
        int small_index = small.front();
        int large_index = large.front();

        small.pop_front();
        large.pop_front();

        out_probas[small_index] = normalized_luminance_of_pixels[small_index];
        out_alias[small_index] = large_index;

        normalized_luminance_of_pixels[large_index] = (normalized_luminance_of_pixels[large_index] + normalized_luminance_of_pixels[small_index]) - 1.0;
        if (normalized_luminance_of_pixels[large_index] > 1.0)
            large.push_back(large_index);
        else
            small.push_back(large_index);
    }

    while (!large.empty())
    {
        int index = large.front();
        large.pop_front();

        out_probas[index] = 1.0;
    }

    while (!small.empty())
    {
        int index = small.front();
        small.pop_front();

        out_probas[index] = 1.0;
    }
}

size_t Image32Bit::byte_size() const
{
    return width * height * sizeof(unsigned char);
}

bool Image32Bit::is_constant_color(float threshold) const
{
    if (width == 0 || height == 0)
        // Incorrect image
        return false;

    std::vector<float> first_pixel_color(channels);
    for (int i = 0; i < channels; i++)
        first_pixel_color[i] = m_pixel_data[i];

    std::atomic<bool> different_pixel_found = false;

    // Comparing the first pixel to all pixels of the texture and returning as soon as we find one
    // that is not within the threshold
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int i = 0; i < channels; i++)
                if (std::abs(first_pixel_color[i] - m_pixel_data[(y * width + x) * channels + i]) > threshold)
                    return false;

    return true;
}

ColorRGB32F* Image32Bit::get_data_as_ColorRGB32F()
{
    return reinterpret_cast<ColorRGB32F*>(m_pixel_data.data());
}

ColorRGB32F Image32Bit::get_pixel_ColorRGB32F(int pixel_index) const
{
    return ColorRGB32F(m_pixel_data[pixel_index * channels + 0], m_pixel_data[pixel_index * channels + 1], m_pixel_data[pixel_index * channels + 2]);
}

ColorRGBA32F* Image32Bit::get_data_as_ColorRGBA32F()
{
    return reinterpret_cast<ColorRGBA32F*>(m_pixel_data.data());
}

ColorRGBA32F Image32Bit::get_pixel_ColorRGBA32F(int pixel_index) const
{
    return ColorRGBA32F(m_pixel_data[pixel_index * channels + 0], m_pixel_data[pixel_index * channels + 1], m_pixel_data[pixel_index * channels + 2], m_pixel_data[pixel_index * channels + 3]);
}

void Image32Bit::free()
{
    m_pixel_data.clear();
    width = 0;
    height = 0;
    channels = 0;
}

Image32Bit3D::Image32Bit3D() 
{
    width = 0;
    height = 0;
    depth = 0;

    channels = 0;
}

Image32Bit3D::Image32Bit3D(const std::vector<Image32Bit> images)
{
    m_images = images;

    width = images[0].width;
    height = images[0].height;
    depth = images.size();

    channels = images[0].channels;
}

ColorRGBA32F Image32Bit3D::sample_rgba32f(float3 uvw) const
{
    // Sampling in repeat mode so we're just keeping the fractional part
    float u = uvw.x;
    if (u != 1.0f)
        // Only doing that if u != 1.0f because if we actually have
        // uv.x == 1.0f, then subtracting static_cast<int>(uv.x) will
        // give us 0.0f even though we actually want 1.0f (which is correct).
        // 
        // Basically, 1.0f gets transformed into 0.0f even though 1.0f is a correct
        // U coordinate which needs not to be wrapped
        u -= static_cast<int>(uvw.x);

    float v = uvw.y;
    if (v != 1.0f)
        // Same for v
        v -= static_cast<int>(uvw.y);

    float w = uvw.z;
    if (w != 1.0f)
        // Same for w
        w -= static_cast<int>(uvw.z);
    

    // For negative UVs, we also want to repeat and we want, for example, 
    // -0.1f to behave as 0.9f
    u = u < 0 ? 1.0f + u : u;
    v = v < 0 ? 1.0f + v : v;
    w = w < 0 ? 1.0f + w : w;

    // Sampling with [0, 0] bottom-left convention
    v = 1.0f - v;

    int x = (u * (width - 1));
    int y = (v * (height - 1));
    int z = (w * (depth - 1));

    ColorRGBA32F out_color;
    for (int i = 0; i < channels; i++)
        out_color[i] = m_images[z][(x + y * width) * channels + i];

    return out_color;
}
