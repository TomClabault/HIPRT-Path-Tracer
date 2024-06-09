/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/OpenImageDenoiser.h"
#include "HIPRT-Orochi/OrochiBuffer.h"

#include <iostream>

OpenImageDenoiser::OpenImageDenoiser()
{
    m_device = nullptr;
    m_denoised_buffer = nullptr;
}

OpenImageDenoiser::OpenImageDenoiser(int width, int height) : m_width(width), m_height(height)
{
    create_device();
    m_denoised_buffer = m_device.newBuffer(sizeof(ColorRGB) * width * height, oidn::Storage::Managed);
    m_input_color_buffer_oidn = m_device.newBuffer(sizeof(ColorRGB) * width * height, oidn::Storage::Managed);
}

void OpenImageDenoiser::set_use_albedo(bool use_albedo)
{
    m_use_albedo = use_albedo;
}

void OpenImageDenoiser::set_use_normals(bool use_normals)
{
    m_use_normals = use_normals;
}

void OpenImageDenoiser::resize(int new_width, int new_height)
{
    if (!check_device())
        return;

    m_width = new_width;
    m_height = new_height;

    m_denoised_buffer = m_device.newBuffer(sizeof(ColorRGB) * new_width * new_height, oidn::Storage::Managed);
    m_input_color_buffer_oidn = m_device.newBuffer(sizeof(ColorRGB) * new_width * new_height, oidn::Storage::Managed);
}

void OpenImageDenoiser::initialize()
{
    create_device();
}

void OpenImageDenoiser::finalize()
{
    if (!check_device())
        return;

    m_beauty_filter = m_device.newFilter("RT");
    m_beauty_filter.setImage("color", m_input_color_buffer_oidn, oidn::Format::Float3, m_width, m_height);
    m_beauty_filter.setImage("output", m_denoised_buffer, oidn::Format::Float3, m_width, m_height);
    m_beauty_filter.set("hdr", true);
    m_beauty_filter.commit();
}

void OpenImageDenoiser::create_device()
{
    // Create an Open Image Denoise device on the GPU depending
    // on whether we're running on an NVIDIA or AMD GPU
    //
    // -1 and nullptr correspond respectively to the default CUDA/HIP device
    // and the default stream
#ifdef OROCHI_ENABLE_CUEW
    m_device = oidn::newDevice(oidn::DeviceType::CUDA);
#else
    m_device = oidn::newDevice(oidn::DeviceType::HIP);
#endif

    const char* errorMessage;
    if (m_device.getError(errorMessage) != oidn::Error::None)
    {

        std::cerr << "There was an error getting the device for denoising with OIDN: " << errorMessage << std::endl;

        return;
    }

    m_device.commit();
}

bool OpenImageDenoiser::check_device()
{
    if (m_device.getHandle() == nullptr)
    {
        std::cerr << "OIDN denoiser's device isn't initialized..." << std::endl;

        return false;
    }

    return true;
}

void OpenImageDenoiser::denoise(std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> data_to_denoise)
{
    if (!check_device())
        return;

    ColorRGB* to_denoise_pointer = data_to_denoise->map();
    OROCHI_CHECK_ERROR(oroMemcpyDtoD(m_input_color_buffer_oidn.getData(), to_denoise_pointer, sizeof(ColorRGB) * m_width * m_height));
    m_beauty_filter.execute();
    data_to_denoise->unmap();
}

void OpenImageDenoiser::copy_denoised_data_to_buffer(std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> buffer)
{
    ColorRGB* buffer_pointer;
    
    buffer_pointer= buffer->map();
    OROCHI_CHECK_ERROR(oroMemcpyDtoD(buffer_pointer, m_denoised_buffer.getData(), sizeof(ColorRGB) * m_width * m_height));
    buffer->unmap();
}
