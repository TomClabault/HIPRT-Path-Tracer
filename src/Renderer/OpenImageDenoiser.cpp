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

void OpenImageDenoiser::set_use_normals(bool use_normals)
{
    m_use_normals = use_normals;
}

void OpenImageDenoiser::set_denoise_normals(bool denoise_normals_or_not)
{
    m_denoise_normals  = denoise_normals_or_not;
}

void OpenImageDenoiser::set_use_albedo(bool use_albedo)
{
    m_use_albedo = use_albedo;
}

void OpenImageDenoiser::set_denoise_albedo(bool denoise_albedo_or_not)
{
    m_denoise_albedo = denoise_albedo_or_not;
}

void OpenImageDenoiser::resize(int new_width, int new_height)
{
    if (!check_valid_state())
        return;

    m_width = new_width;
    m_height = new_height;

    m_denoised_buffer = m_device.newBuffer(sizeof(ColorRGB32F) * new_width * new_height, oidn::Storage::Managed);
    m_input_color_buffer_oidn = m_device.newBuffer(sizeof(ColorRGB32F) * new_width * new_height, oidn::Storage::Managed);
}

void OpenImageDenoiser::initialize()
{
    create_device();
}

void OpenImageDenoiser::finalize()
{
    if (!check_valid_state())
        return;

    m_beauty_filter = m_device.newFilter("RT");
    m_beauty_filter.setImage("color", m_input_color_buffer_oidn, oidn::Format::Float3, m_width, m_height);
    m_beauty_filter.setImage("output", m_denoised_buffer, oidn::Format::Float3, m_width, m_height);
    m_beauty_filter.set("cleanAux", m_denoise_albedo && m_denoise_normals);
    m_beauty_filter.set("hdr", true);

    if (m_use_normals)
    {
        // Creating the buffers here instead of in resize() because we want the creation/destruction 
        // to be dynamic in response to ImGui input so we cannot just wait for a window queue_resize event
        // that would trigger OpenImageDenoiser::queue_resize()
        m_normals_buffer_denoised_oidn = m_device.newBuffer(sizeof(float3) * m_width * m_height);
        
        m_beauty_filter.setImage("normal", m_normals_buffer_denoised_oidn, oidn::Format::Float3, m_width, m_height);
    }
    else
        // Destroying the filter & buffer to free memory
        m_normals_buffer_denoised_oidn = nullptr;

    if (m_denoise_normals && m_use_normals)
    {
        m_normals_filter = m_device.newFilter("RT");
        m_normals_filter.setImage("normal", m_normals_buffer_denoised_oidn, oidn::Format::Float3, m_width, m_height);
        m_normals_filter.setImage("output", m_normals_buffer_denoised_oidn, oidn::Format::Float3, m_width, m_height);
        m_normals_filter.commit();
    }
    else
        m_normals_filter = nullptr;

    if (m_use_albedo)
    {
        // Creating the buffers here instead of in resize() because we want the creation/destruction 
        // to be dynamic in response to ImGui input so we cannot just wait for a window queue_resize event
        // that would trigger OpenImageDenoiser::queue_resize()
        m_albedo_buffer_denoised_oidn = m_device.newBuffer(sizeof(ColorRGB32F) * m_width * m_height, oidn::Storage::Managed);

        m_beauty_filter.setImage("albedo", m_albedo_buffer_denoised_oidn, oidn::Format::Float3, m_width, m_height);
    }
    else
        // Destroying the filter & buffer to free memory
        m_albedo_buffer_denoised_oidn = nullptr;

    if (m_denoise_albedo && m_use_albedo)
    {
        m_albedo_filter = m_device.newFilter("RT");
        m_albedo_filter.setImage("albedo", m_albedo_buffer_denoised_oidn, oidn::Format::Float3, m_width, m_height);
        m_albedo_filter.setImage("output", m_albedo_buffer_denoised_oidn, oidn::Format::Float3, m_width, m_height);
        m_albedo_filter.commit();
    }
    else
        m_albedo_filter = nullptr;

    m_beauty_filter.commit();
}

void OpenImageDenoiser::create_device()
{
    // Create an Open ImageRGB32F Denoise device on the GPU depending
    // on whether we're running on an NVIDIA or AMD GPU
    //
    // -1 and nullptr correspond respectively to the default CUDA/HIP device
    // and the default stream
#ifdef OROCHI_ENABLE_CUEW
    m_device = oidn::newDevice(oidn::DeviceType::CUDA);
#else
    m_device = oidn::newDevice(oidn::DeviceType::HIP);
#endif

    if (m_device.getError() == oidn::Error::UnsupportedHardware)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Could not create an OIDN GPU device. Falling back to CPU...");

        m_device = oidn::newDevice(oidn::DeviceType::CPU);

        const char* errorMessage;
        if (m_device.getError(errorMessage) != oidn::Error::None)
        {
            g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "There was an error getting a CPU device for denoising with OIDN. Denoiser will be unavailable. %s", errorMessage);

            m_denoiser_invalid = true;
            return;
        }
        else
            // Valid creation of a CPU device
            m_cpu_device = true;
    }

    bool managedMemory = m_device.get<bool>("managedMemorySupported");

    m_device.commit();
}

bool OpenImageDenoiser::check_valid_state()
{
    if (m_denoiser_invalid)
        // Returning false without error message, the error was already printed when we failed at creating the device
        return false;
    else if (!check_device())
        // check_device prints the error
        return false;

    return true;
}

bool OpenImageDenoiser::check_device()
{
    if (m_device.getHandle() == nullptr)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "OIDN denoiser's device isn't initialized...");

        return false;
    }

    return true;
}

bool OpenImageDenoiser::check_buffer_sizes()
{
    size_t normals_buffer_size = m_normals_buffer_denoised_oidn.getSize() / sizeof(float3);
    size_t albedo_buffer_size = m_albedo_buffer_denoised_oidn.getSize() / sizeof(ColorRGB32F);
    size_t denoised_buffer_size = m_denoised_buffer.getSize() / sizeof(ColorRGB32F);
    size_t noisy_input_buffer_size = m_input_color_buffer_oidn.getSize() / sizeof(ColorRGB32F);

    if (m_use_normals && normals_buffer_size != m_width * m_height)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "The denoiser normals buffer isn't the same size as the denoiser. Did you forget to call finalize() after a call to resize()?");

        return false;
    }
    else if (m_use_albedo && albedo_buffer_size != m_width * m_height)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "The denoiser albedo buffer isn't the same size as the denoiser. Did you forget to call finalize() after a call to resize()?");

        return false;
    }
    else if (denoised_buffer_size != m_width * m_height || noisy_input_buffer_size != m_width * m_height)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "The denoiser output buffer or input noisy buffer isn't the same size as the denoiser. This has to be an internal error since resize() resizes these two buffers.");

        return false;
    }

    return true;
}

void OpenImageDenoiser::denoise(std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> data_to_denoise, std::shared_ptr<OpenGLInteropBuffer<float3>> normals_aov, std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> albedo_aov)
{
    if (!check_valid_state())
        return;

    if (!check_buffer_sizes())
        return;

    oroMemcpyKind memcpyKind = m_cpu_device ? oroMemcpyDeviceToHost : oroMemcpyDeviceToDevice;

    if (normals_aov != nullptr)
    {
        float3* normals_pointer = normals_aov->map_no_error();
        OROCHI_CHECK_ERROR(oroMemcpy(m_normals_buffer_denoised_oidn.getData(), normals_pointer, sizeof(float3) * m_width * m_height, memcpyKind));
        normals_aov->unmap();

        if (m_denoise_normals)
            m_normals_filter.execute();
    }

    if (albedo_aov != nullptr)
    {
        ColorRGB32F* albedo_pointer = albedo_aov->map_no_error();
        OROCHI_CHECK_ERROR(oroMemcpy(m_albedo_buffer_denoised_oidn.getData(), albedo_pointer, sizeof(ColorRGB32F) * m_width * m_height, memcpyKind));
        albedo_aov->unmap();

        if (m_denoise_albedo)
            m_albedo_filter.execute();
    }
    
    ColorRGB32F* data_to_denoise_pointer = data_to_denoise->map_no_error();
    OROCHI_CHECK_ERROR(oroMemcpy(m_input_color_buffer_oidn.getData(), data_to_denoise_pointer, sizeof(ColorRGB32F) * m_width * m_height, memcpyKind));
    m_beauty_filter.execute();
    data_to_denoise->unmap();
}

void OpenImageDenoiser::copy_denoised_data_to_buffer(std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> out_buffer)
{
    oroMemcpyKind memcpyKind;
    ColorRGB32F* buffer_pointer;
    
    memcpyKind = m_cpu_device ? oroMemcpyHostToDevice : oroMemcpyDeviceToDevice;
    buffer_pointer= out_buffer->map();
    OROCHI_CHECK_ERROR(oroMemcpy(buffer_pointer, m_denoised_buffer.getData(), sizeof(ColorRGB32F) * m_width * m_height, memcpyKind));
    out_buffer->unmap();
}
