/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Utils/Utils.h"

Image::Image(const std::string& filepath) : Image(filepath.c_str()) {}

Image::Image(const char* filepath)
{
    *this = Image::read_image(filepath, false);
}

Image Image::read_image(const std::string& filepath, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &channels, 3);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    Image output_image(width, height);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;
            ColorRGB float_pixel_color;
            float_pixel_color.r = pixels[index * 3 + 0] / 255.0f;
            float_pixel_color.g = pixels[index * 3 + 1] / 255.0f;
            float_pixel_color.b = pixels[index * 3 + 2] / 255.0f;

            output_image[index] = float_pixel_color;
        }
    }

    output_image.channels = 3;
    return output_image;
}

Image Image::read_image_hdr(const std::string& filepath, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &channels, 3);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    Image image(reinterpret_cast<ColorRGB*>(pixels), width, height);
    image.channels = 3;

    return image;
}

bool Image::write_image_png(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<unsigned char> tmp(width * height * channels);
    for (unsigned i = 0; i < width * height; i++)
    {
        ColorRGB pixel = m_pixel_data[i] * 255;

        tmp[i * 3 + 0] = hippt::clamp(0.0f, 255.0f, pixel.r);
        tmp[i * 3 + 1] = hippt::clamp(0.0f, 255.0f, pixel.g);
        tmp[i * 3 + 2] = hippt::clamp(0.0f, 255.0f, pixel.b);
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





ImageRGBA::ImageRGBA(const std::string& filepath) : ImageRGBA(filepath.c_str()) {}

ImageRGBA::ImageRGBA(const char* filepath)
{
    *this = ImageRGBA::read_image(filepath, false);
}

ImageRGBA ImageRGBA::read_image(const std::string& filepath, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, channels;
    unsigned char* pixels = stbi_load(filepath.c_str(), &width, &height, &channels, 4);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    ImageRGBA output_image(width, height);
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = x + y * width;
            ColorRGBA float_pixel_color;
            float_pixel_color.r = pixels[index * 4 + 0] / 255.0f;
            float_pixel_color.g = pixels[index * 4 + 1] / 255.0f;
            float_pixel_color.b = pixels[index * 4 + 2] / 255.0f;
            float_pixel_color.a = pixels[index * 4 + 3] / 255.0f;

            output_image[index] = float_pixel_color;
        }
    }

    output_image.channels = 4;
    return output_image;
}

ImageRGBA ImageRGBA::read_image_hdr(const std::string& filepath, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int width, height, channels;
    float* pixels = stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);

    if (!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        Utils::debugbreak();

        std::exit(1);
    }

    ImageRGBA image(reinterpret_cast<ColorRGBA*>(pixels), width, height);
    image.channels = 4;

    return image;
}

bool ImageRGBA::write_image_png(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    std::vector<unsigned char> tmp(byte_size());
    for (unsigned i = 0; i < width * height; i++)
    {
        ColorRGBA pixel = m_pixel_data[i] * 255;

        tmp[i * 4 + 0] = hippt::clamp(0.0f, 255.0f, pixel.r);
        tmp[i * 4 + 1] = hippt::clamp(0.0f, 255.0f, pixel.g);
        tmp[i * 4 + 2] = hippt::clamp(0.0f, 255.0f, pixel.b);
        tmp[i * 4 + 3] = hippt::clamp(0.0f, 255.0f, pixel.a);
    }

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_png(filename, width, height, 4, tmp.data(), width * 4) != 0;
}

bool ImageRGBA::write_image_hdr(const char* filename, const bool flipY) const
{
    if (byte_size() == 0)
        return false;

    stbi_flip_vertically_on_write(flipY);
    return stbi_write_hdr(filename, width, height, 4, reinterpret_cast<const float*>(m_pixel_data.data())) != 0;
}
