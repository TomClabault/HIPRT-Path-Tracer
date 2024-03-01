//Already defined in image_io.cpp from gkit
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Image/image.h"
#include "Utils/utils.h"

#include <iostream>
#include <string>

#include <OpenImageDenoise/oidn.hpp>

Image Utils::read_image_float(const std::string& filepath, int& image_width, int& image_height, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int channels;
    float* pixels = stbi_loadf(filepath.c_str(), &image_width, &image_height, &channels, 0);

    if(!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        std::exit(1);
    }

    return Image(reinterpret_cast<Color*>(pixels), image_width, image_height);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const Image& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(reinterpret_cast<const float*>(hdr_image.data().data()), hdr_image.width * hdr_image.height, sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<Color>& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(reinterpret_cast<const float*>(hdr_image.data()), hdr_image.size(), sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(hdr_image.data(), hdr_image.size() / 3, sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const float* hdr_image, size_t pixel_count, int sample_number, float gamma, float exposure)
{
    std::vector<unsigned char> tonemapped_data(pixel_count * 3);

#pragma omp parallel for
    for (int i = 0; i < pixel_count; i += 3)
    {
        Color pixel = Color(hdr_image[i + 0], hdr_image[i + 1], hdr_image[i + 2]) / (float)sample_number;
        Color tone_mapped = Color(1.0f, 1.0f, 1.0f) - exp(-pixel * exposure);
        Color gamma_corrected = pow(tone_mapped, 1.0f / gamma);

        tonemapped_data[i + 0] = gamma_corrected.r * 255.0f;
        tonemapped_data[i + 1] = gamma_corrected.g * 255.0f;
        tonemapped_data[i + 2] = gamma_corrected.b * 255.0f;
    }

    return tonemapped_data;
}

Image Utils::OIDN_denoise(const Image& image, int width, int height, float blend_factor)
{
    // Create an Open Image Denoise device
    static bool device_done = false;
    static oidn::DeviceRef device;
    if (!device_done)
    {
        device = oidn::newDevice(); // CPU or GPU if available
        if (device == NULL)
        {
            std::cerr << "There was an error getting the device for denoising with OIDN. Perhaps some missing DLLs for your hardware?" << std::endl;
            return Image();
        }
        device.commit();

        device_done = true;
    }

    // Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
    oidn::BufferRef colorBuf = device.newBuffer(width * height * 3 * sizeof(float));
    // Create a filter for denoising a beauty (color) image using optional auxiliary images too
    // This can be an expensive operation, so try no to create a new filter for every image!
    static oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color", colorBuf, oidn::Format::Float3, width, height); // beauty
    filter.setImage("output", colorBuf, oidn::Format::Float3, width, height); // denoised beauty
    filter.set("hdr", true); // beauty image is HDR
    filter.commit();
    // Fill the input image buffers
    float* colorPtr = (float*)colorBuf.getData();
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            colorPtr[index * 3 + 0] = image[index].r;
            colorPtr[index * 3 + 1] = image[index].g;
            colorPtr[index * 3 + 2] = image[index].b;
        }
    // Filter the beauty image

    filter.execute();

    float* denoised_ptr = (float*)colorBuf.getData();
    Image output_image(width, height);
    std::vector<Color>& output_pixels = output_image.data();
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            Color color = blend_factor * Color(denoised_ptr[index * 3 + 0], denoised_ptr[index * 3 + 1], denoised_ptr[index * 3 + 2])
                + (1.0f - blend_factor) * image[index];

            output_pixels[index] = color;
        }

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return output_image;
}
