#include "Renderer/open_image_denoiser.h"

#include <iostream>

OpenImageDenoiser::OpenImageDenoiser(Color* color_buffer) : m_color_buffer(color_buffer)
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

OpenImageDenoiser::OpenImageDenoiser(Color* color_buffer, hiprtFloat3* world_space_normals_buffer) : OpenImageDenoiser(color_buffer)
{
    m_use_normals = true;

    m_normals_buffer = world_space_normals_buffer;
}

OpenImageDenoiser::OpenImageDenoiser(Color* color_buffer, Color* albedo_buffer) : OpenImageDenoiser(color_buffer)
{
    m_use_albedo = true;

    m_albedo_buffer = albedo_buffer;
}

OpenImageDenoiser::OpenImageDenoiser(Color* color_buffer, hiprtFloat3* world_space_normals_buffer, Color* albedo_buffer) : OpenImageDenoiser(color_buffer)
{
    m_use_albedo = true;
    m_use_normals = true;

    m_albedo_buffer = albedo_buffer;
    m_normals_buffer = world_space_normals_buffer;
}

void OpenImageDenoiser::resize(int new_width, int new_height, Color* color_buffer, hiprtFloat3* normals_buffer, Color* albedo_buffer)
{
    m_color_buffer = color_buffer;
    m_normals_buffer = normals_buffer;
    m_albedo_buffer = albedo_buffer;

    m_denoised_buffer = m_device.newBuffer(new_width * new_height * 3 * sizeof(float));

    m_width = new_width;
    m_height = new_height;

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
    {
        std::cout << "Buffer configuration error: " << errorMessage << std::endl;
        return;
    }

    create_filters(new_width, new_height);
}

void OpenImageDenoiser::create_filters(int width, int height)
{
    m_beauty_filter = m_device.newFilter("RT"); // generic ray tracing filter
    m_beauty_filter.setImage("color", m_color_buffer, oidn::Format::Float3, width, height); // beauty
    m_beauty_filter.setImage("albedo", m_albedo_buffer, oidn::Format::Float3, width, height); // albedo aov
    m_beauty_filter.setImage("normal", m_normals_buffer, oidn::Format::Float3, width, height); // normals aov
    m_beauty_filter.setImage("output", m_denoised_buffer, oidn::Format::Float3, width, height); // denoised beauty
    m_beauty_filter.set("cleanAux", true); // Normals and albedo are not noisy
    m_beauty_filter.set("hdr", true); // beauty image is HDR
    m_beauty_filter.commit();

    if (m_use_albedo)
    {
        m_albedo_filter = m_device.newFilter("RT");
        m_albedo_filter.setImage("albedo", m_albedo_buffer, oidn::Format::Float3, width, height);
        m_albedo_filter.setImage("output", m_albedo_buffer, oidn::Format::Float3, width, height);
        m_albedo_filter.commit();
    }

    if (m_use_normals)
    {
        m_normals_filter = m_device.newFilter("RT");
        m_normals_filter.setImage("normal", m_normals_buffer, oidn::Format::Float3, width, height);
        m_normals_filter.setImage("output", m_normals_buffer, oidn::Format::Float3, width, height);
        m_normals_filter.commit();
    }

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
    {
        std::cout << "Filter configuration error: " << errorMessage << std::endl;
        return;
    }
}

void OpenImageDenoiser::denoise()
{
    // Fill the input image buffers
    if (m_use_albedo)
        m_albedo_filter.execute();
    if (m_use_normals)
        m_normals_filter.execute();

    m_beauty_filter.execute();

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;
}

std::vector<Color> OpenImageDenoiser::get_denoised_data()
{
    Color* denoised_ptr = (Color*)m_denoised_buffer.getData();

    std::vector<Color> denoised_output;
    denoised_output.insert(denoised_output.end(), &denoised_ptr[0], &denoised_ptr[m_width * m_height]);

    return denoised_output;
}

void* OpenImageDenoiser::get_denoised_data_pointer()
{
    return m_denoised_buffer.getData();
}
