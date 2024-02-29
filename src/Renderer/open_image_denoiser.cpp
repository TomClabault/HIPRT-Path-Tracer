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

OpenImageDenoiser::OpenImageDenoiser(hiprtFloat3* world_space_normals_buffer) : OpenImageDenoiser()
{
    m_use_normals = true;
    m_normals_buffer = world_space_normals_buffer;
}

OpenImageDenoiser::OpenImageDenoiser(Color* albedo_buffer) : OpenImageDenoiser()
{
    m_use_albedo = true;
    m_albedo_buffer = albedo_buffer;
}

OpenImageDenoiser::OpenImageDenoiser(hiprtFloat3* world_space_normals_buffer, Color* albedo_buffer) : OpenImageDenoiser()
{
    m_use_albedo = true;
    m_use_normals = true;

    m_albedo_buffer = albedo_buffer;
    m_normals_buffer = world_space_normals_buffer;
}

void OpenImageDenoiser::resize_buffers(int new_width, int new_height)
{
    //m_color_buffer = m_device.newBuffer(new_width * new_height * 3 * sizeof(float));

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
    m_beauty_filter.set("cleanAux", true); // Normals and albedo are not noisy
    m_beauty_filter.set("hdr", true); // beauty image is HDR
    m_beauty_filter.commit();

    if (m_use_albedo)
    {
        m_albedo_filter = m_device.newFilter("RT");
        m_albedo_filter.setImage("albedo", m_albedo_buffer, oidn::Format::Float3, new_width, new_height);
        m_albedo_filter.setImage("output", m_albedo_buffer, oidn::Format::Float3, new_width, new_height);
        m_albedo_filter.commit();
    }

    if (m_use_normals)
    {
        m_normals_filter = m_device.newFilter("RT");
        m_normals_filter.setImage("normal", m_normals_buffer, oidn::Format::Float3, new_width, new_height);
        m_normals_filter.setImage("output", m_normals_buffer, oidn::Format::Float3, new_width, new_height);
        m_normals_filter.commit();
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

    if (m_use_albedo)
        m_albedo_filter.execute();
    if (m_use_normals)
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
