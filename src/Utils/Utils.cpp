/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "stb_image.h"

#include "Image/Image.h"
#include "Utils/Utils.h"

#include <iostream>
#include <OpenImageDenoise/oidn.hpp>
#include <string>
#include <sstream>

std::vector<unsigned char> Utils::tonemap_hdr_image(const Image& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(reinterpret_cast<const float*>(hdr_image.data().data()), hdr_image.width * hdr_image.height * 3, sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<ColorRGB>& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(reinterpret_cast<const float*>(hdr_image.data()), hdr_image.size() * 3, sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(hdr_image.data(), hdr_image.size(), sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const float* hdr_image, size_t float_count, int sample_number, float gamma, float exposure)
{
    std::vector<unsigned char> tonemapped_data(float_count);

#pragma omp parallel for
    for (int i = 0; i < float_count; i += 3)
    {
        ColorRGB pixel = ColorRGB(hdr_image[i + 0], hdr_image[i + 1], hdr_image[i + 2]) / (float)sample_number;
        ColorRGB tone_mapped = ColorRGB(1.0f, 1.0f, 1.0f) - exp(-pixel * exposure);
        ColorRGB gamma_corrected = pow(tone_mapped, 1.0f / gamma);

        tonemapped_data[i + 0] = gamma_corrected.r * 255.0f;
        tonemapped_data[i + 1] = gamma_corrected.g * 255.0f;
        tonemapped_data[i + 2] = gamma_corrected.b * 255.0f;
    }

    return tonemapped_data;
}

std::string Utils::file_to_string(const char* filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file " << filepath << std::endl;

        return std::string();
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

Image Utils::OIDN_denoise(const Image& image, int width, int height, float blend_factor)
{
    // Create an Open Image Denoise device
    static bool device_done = false;
    static oidn::DeviceRef device;
    if (!device_done)
    {
        // We're going to create a CPU device as there seems to be some issues with the GPU (HIP at least)
        // device on Linux
        int num_devices = oidnGetNumPhysicalDevices();
        for (int i = 0; i < num_devices; i++)
        {
            if (static_cast<oidn::DeviceType>(oidnGetPhysicalDeviceInt(i, "type")) == oidn::DeviceType::CPU)
            {
                device = oidn::newDevice(i);
                if (device.getHandle() == nullptr)
                {
                    std::cerr << "There was an error getting the device for denoising with OIDN. Perhaps some missing libraries for your hardware?" << std::endl;
                    return Image();
                }
                device.commit();

                device_done = true;
            }
        }
    }

    if (!device_done)
    {
        // If we couldn't make a CPU device, trying GPU
        device = oidn::newDevice();
        if (device.getHandle() == nullptr)
        {
            std::cerr << "There was an error getting the device for denoising with OIDN. Perhaps some missing libraries for your hardware?" << std::endl;
            return Image();
        }
        device.commit();

        device_done = true;
    }

    if (!device_done)
    {
        std::cerr << "Cannot create any OIDN device, aborting denoising..." << std::endl;
        return Image(1, 1);
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
    std::vector<ColorRGB>& output_pixels = output_image.data();
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            ColorRGB color = blend_factor * ColorRGB(denoised_ptr[index * 3 + 0], denoised_ptr[index * 3 + 1], denoised_ptr[index * 3 + 2])
                + (1.0f - blend_factor) * image[index];

            output_pixels[index] = color;
        }

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return output_image;
}
