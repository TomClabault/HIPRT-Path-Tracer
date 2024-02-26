#include "Renderer/open_image_denoiser.h"

#include <iostream>

OpenImageDenoiser::OpenImageDenoiser()
{
    // Create an Open Image Denoise device
    m_device = oidn::newDevice(); // CPU or GPU if available
    if (m_device == NULL)
    {
        std::cerr << "There was an error getting the device for denoising with OIDN. Perhaps some missing DLLs for your hardware?" << std::endl;

        return;
    }

    m_device.commit();
}

void OpenImageDenoiser::resize_buffers(int new_width, int new_height)
{
    m_color_buffer = m_device.newBuffer(new_width * new_height * 3 * sizeof(float));
    m_albedo_buffer = m_device.newBuffer(new_width * new_height * 3 * sizeof(float));
    m_normals_buffer = m_device.newBuffer(new_width * new_height * 3 * sizeof(float));

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
    {
        std::cout << "Buffer configuration error: " << errorMessage << std::endl;
        return;
    }

    m_filter = m_device.newFilter("RT"); // generic ray tracing filter
    m_filter.setImage("color", m_color_buffer, oidn::Format::Float3, new_width, new_height); // beauty
    m_filter.setImage("albedo", m_albedo_buffer, oidn::Format::Float3, new_width, new_height); // albedo aov
    m_filter.setImage("normal", m_normals_buffer, oidn::Format::Float3, new_width, new_height); // normals aov
    m_filter.setImage("output", m_color_buffer, oidn::Format::Float3, new_width, new_height); // denoised beauty
    m_filter.set("hdr", true); // beauty image is HDR
    m_filter.commit();

    if (m_device.getError(errorMessage) != oidn::Error::None)
    {
        std::cout << "Filter configuration error: " << errorMessage << std::endl;
        return;
    }
}

std::vector<float> OpenImageDenoiser::denoise(int width, int height, const std::vector<float>& to_denoise)
{
    // Fill the input image buffers
    float* colorPtr = (float*)m_color_buffer.getData();

    //std::memcpy(colorPtr, to_denoise.data(), to_denoise.size() * sizeof(float));
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            colorPtr[index * 3 + 0] = to_denoise[index * 3 + 0];
            colorPtr[index * 3 + 1] = to_denoise[index * 3 + 1];
            colorPtr[index * 3 + 2] = to_denoise[index * 3 + 2];
        }

    // Filter the beauty image
    m_filter.execute();

    float* denoised_ptr = (float*)m_color_buffer.getData();
    std::vector<float> denoised_output(to_denoise.size());
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            /*Color color = blend_factor * Color(denoised_ptr[index * 3 + 0], denoised_ptr[index * 3 + 1], denoised_ptr[index * 3 + 2])
                + (1.0f - blend_factor) * image[index];
            color.a = 1.0f;*/

            denoised_output[index * 3 + 0] = denoised_ptr[index * 3 + 0];
            denoised_output[index * 3 + 1] = denoised_ptr[index * 3 + 1];
            denoised_output[index * 3 + 2] = denoised_ptr[index * 3 + 2];
        }

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return denoised_output;
}

std::vector<float> OpenImageDenoiser::denoise(int width, int height, const std::vector<float>& to_denoise, const std::vector<hiprtFloat3>& world_space_normals_aov_buffer, const std::vector<Color>& albedo_aov_buffer)
{
    // Fill the input image buffers
    float* colorPtr = (float*)m_color_buffer.getData();
    //std::memcpy(colorPtr, to_denoise.data(), to_denoise.size() * sizeof(float));
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            colorPtr[index * 3 + 0] = to_denoise[index * 3 + 0];
            colorPtr[index * 3 + 1] = to_denoise[index * 3 + 1];
            colorPtr[index * 3 + 2] = to_denoise[index * 3 + 2];
        }
    }

    float* albedoPtr = (float*)m_albedo_buffer.getData();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            albedoPtr[index * 3 + 0] = albedo_aov_buffer[index].r;
            albedoPtr[index * 3 + 1] = albedo_aov_buffer[index].g;
            albedoPtr[index * 3 + 2] = albedo_aov_buffer[index].b;
        }
    }

    float* normalsPtr = (float*)m_normals_buffer.getData();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            normalsPtr[index * 3 + 0] = world_space_normals_aov_buffer[index].x;
            normalsPtr[index * 3 + 1] = world_space_normals_aov_buffer[index].y;
            normalsPtr[index * 3 + 2] = world_space_normals_aov_buffer[index].z;
        }
    }

    // Filter the beauty image
    m_filter.execute();

    float* denoised_ptr = (float*)m_color_buffer.getData();
    std::vector<float> denoised_output(to_denoise.size());
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            /*Color color = blend_factor * Color(denoised_ptr[index * 3 + 0], denoised_ptr[index * 3 + 1], denoised_ptr[index * 3 + 2])
                + (1.0f - blend_factor) * image[index];
            color.a = 1.0f;*/

            denoised_output[index * 3 + 0] = denoised_ptr[index * 3 + 0];
            denoised_output[index * 3 + 1] = denoised_ptr[index * 3 + 1];
            denoised_output[index * 3 + 2] = denoised_ptr[index * 3 + 2];
        }

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return denoised_output;
}
