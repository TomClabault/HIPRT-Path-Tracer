/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/OpenImageDenoiser.h"

#include <iostream>

OpenImageDenoiser::OpenImageDenoiser()
{
    m_device = nullptr;
    m_denoiser_buffer = nullptr;
}

OpenImageDenoiser::OpenImageDenoiser(int width, int height) : m_width(width), m_height(height), m_denoiser_buffer(std::make_shared<OpenGLInteropBuffer<ColorRGB>>())
{
    create_device();
    m_denoiser_buffer = std::make_shared<OpenGLInteropBuffer<ColorRGB>>();
    m_denoiser_buffer->resize(width * height);
}

void OpenImageDenoiser::set_use_albedo(bool use_albedo)
{
    m_use_albedo = use_albedo;
}

void OpenImageDenoiser::set_use_normals(bool use_normals)
{
    m_use_normals = use_normals;
}

void OpenImageDenoiser::set_color_buffer(std::shared_ptr<OpenGLInteropBuffer<ColorRGB>>color_buffer)
{
    m_input_color_buffer = std::weak_ptr<OpenGLInteropBuffer<ColorRGB>>(color_buffer);
}

void OpenImageDenoiser::resize(int new_width, int new_height)
{
    m_width = new_width;
    m_height = new_height;

    if (m_denoiser_buffer == nullptr)
        m_denoiser_buffer = std::make_shared<OpenGLInteropBuffer<ColorRGB>>();
    m_denoiser_buffer->resize(m_width * m_height);
}

void OpenImageDenoiser::finalize()
{
    if (m_device.getHandle() == nullptr)
        create_device();

    if (!check_denoiser_validity())
        return;

    std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> color_buffer_device = acquire_input_color_buffer();
    if (color_buffer_device == nullptr)
        return;

    m_beauty_filter = m_device.newFilter("RT");
    m_beauty_filter.setImage("color", reinterpret_cast<void*>(color_buffer_device->map()), oidn::Format::Float3, m_width, m_height);
    m_beauty_filter.setImage("output", m_denoiser_buffer->map(), oidn::Format::Float3, m_width, m_height);
    m_beauty_filter.set("hdr", true);
    m_beauty_filter.commit();

    color_buffer_device->unmap();
    m_denoiser_buffer->unmap();
}

void OpenImageDenoiser::create_device()
{
    // Create an Open Image Denoise device on the GPU depending
    // on whether we're running on an NVIDIA or AMD GPU
    //
    // -1 and nullptr correspond respectively to the default CUDA/HIP device
    // and the default stream
#ifdef OROCHI_ENABLE_CUEW
    m_device = oidn::newCUDADevice(-1, nullptr);
#else
    m_device = oidn::newHIPDevice(-1, nullptr);
#endif

    if (m_device.getHandle() == nullptr)
    {
        std::cerr << "There was an error getting the device for denoising with OIDN. Perhaps some missing librariries for your hardware?" << std::endl;

        return;
    }

    m_device.commit();
}

bool OpenImageDenoiser::check_denoiser_validity()
{
    std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> color_buffer_device = acquire_input_color_buffer();
    if (color_buffer_device == nullptr)
        return false;

    // Checking that the input color buffer and the denoised buffer are the same size
    size_t input_color_buffer_element_count = color_buffer_device->get_element_count();
    size_t denoiser_buffer_element_count = m_denoiser_buffer->get_element_count();
    if (denoiser_buffer_element_count != input_color_buffer_element_count)
    {
        std::cerr << "The buffer for the denoiser image and the input device color buffer are not the same size, respectively: " 
            << denoiser_buffer_element_count << " and " << input_color_buffer_element_count << " elements large." << std::endl;
    }

        return true;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> OpenImageDenoiser::acquire_input_color_buffer()
{
    std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> color_buffer_device = m_input_color_buffer.lock();

    if (!color_buffer_device)
    {
        std::cerr << "The input color buffer of the denoiser is NULL. Cannot finalize()." << std::endl;

        return nullptr;
    }

    return color_buffer_device;
}

void OpenImageDenoiser::denoise()
{
    m_beauty_filter.execute();
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB>> OpenImageDenoiser::get_denoised_buffer()
{
    return m_denoiser_buffer;
}

