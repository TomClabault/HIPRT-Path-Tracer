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

void OpenImageDenoiser::resize_buffers(int new_width, int new_height, bool use_AOVs)
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

    m_beauty_filter = m_device.newFilter("RT"); // generic ray tracing filter
    m_beauty_filter.setImage("color", m_color_buffer, oidn::Format::Float3, new_width, new_height); // beauty
    m_beauty_filter.setImage("albedo", m_albedo_buffer, oidn::Format::Float3, new_width, new_height); // albedo aov
    m_beauty_filter.setImage("normal", m_normals_buffer, oidn::Format::Float3, new_width, new_height); // normals aov
    m_beauty_filter.setImage("output", m_color_buffer, oidn::Format::Float3, new_width, new_height); // denoised beauty
    m_beauty_filter.set("cleanAux", use_AOVs); // Normals and albedo are not noisy
    m_beauty_filter.set("hdr", true); // beauty image is HDR
    m_beauty_filter.commit();

    if (use_AOVs)
    {
        m_albedo_filter = m_device.newFilter("RT");
        m_albedo_filter.setImage("albedo", m_albedo_buffer, oidn::Format::Float3, new_width, new_height);
        m_albedo_filter.setImage("output", m_albedo_buffer, oidn::Format::Float3, new_width, new_height);
        m_albedo_filter.commit();

        m_normals_filter = m_device.newFilter("RT");
        m_normals_filter.setImage("normal", m_normals_buffer, oidn::Format::Float3, new_width, new_height);
        m_normals_filter.setImage("output", m_normals_buffer, oidn::Format::Float3, new_width, new_height);
        m_normals_filter.commit();
    }
    else
    {
        m_albedo_filter.release();
        m_normals_filter.release();
    }

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
    std::memcpy(colorPtr, to_denoise.data(), to_denoise.size() * sizeof(float));

    // Filter the beauty image
    m_beauty_filter.execute();

    float* denoised_ptr = (float*)m_color_buffer.getData();
    std::vector<float> denoised_output;
    denoised_output.insert(denoised_output.end(), &denoised_ptr[0], &denoised_ptr[to_denoise.size()]);

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return denoised_output;
}

std::vector<float> OpenImageDenoiser::denoise(int width, int height, const std::vector<float>& to_denoise, const std::vector<hiprtFloat3>& world_space_normals_aov_buffer, const std::vector<Color>& albedo_aov_buffer)
{
    // Fill the input image buffers
    float* colorPtr = (float*)m_color_buffer.getData();
    float* albedoPtr = (float*)m_albedo_buffer.getData();
    float* normalsPtr = (float*)m_normals_buffer.getData();

    std::memcpy(colorPtr, to_denoise.data(), to_denoise.size() * sizeof(float));
    std::memcpy(albedoPtr, albedo_aov_buffer.data(), albedo_aov_buffer.size() * sizeof(float));
    std::memcpy(normalsPtr, world_space_normals_aov_buffer.data(), world_space_normals_aov_buffer.size() * sizeof(float));

    m_albedo_filter.execute();
    m_normals_filter.execute();
    m_beauty_filter.execute();

    float* denoised_ptr = (float*)m_color_buffer.getData();
    std::vector<float> denoised_output;
    denoised_output.insert(denoised_output.end(), &denoised_ptr[0], &denoised_ptr[to_denoise.size()]);

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return denoised_output;
}
