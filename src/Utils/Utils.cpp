/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "stb_image.h"

#include "Image/Image.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

#include <iostream>
#include <iomanip> // get_current_date_string()
#include <OpenImageDenoise/oidn.hpp>
#include <string>
#include <sstream>

#if defined(_WIN32) || defined(_WIN32_WCE) || defined(__WIN32__)
#include <Windows.h> // for is_file_on_SSD()
#endif

extern ImGuiLogger g_imgui_logger;

std::vector<unsigned char> Utils::tonemap_hdr_image(const Image32Bit& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(reinterpret_cast<const float*>(hdr_image.data().data()), hdr_image.width * hdr_image.height * 3, sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<ColorRGB32F>& hdr_image, int sample_number, float gamma, float exposure)
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
        ColorRGB32F pixel = ColorRGB32F(hdr_image[i + 0], hdr_image[i + 1], hdr_image[i + 2]) / static_cast<float>(sample_number);
        ColorRGB32F tone_mapped = ColorRGB32F(1.0f, 1.0f, 1.0f) - exp(-pixel * exposure);
        ColorRGB32F gamma_corrected = pow(tone_mapped, 1.0f / gamma);

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
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Unable to open file %s", filepath);

        return std::string();
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

void Utils::get_current_date_string(std::stringstream& ss)
{
	std::time_t t = std::time(0);
	std::tm* now = std::localtime(&t);

	ss << std::put_time(now, "%m.%d.%Y.%H.%M.%S");
}

void* Utils::get_volume_handle_for_file(const char* filePath)
{
#if !defined(_WIN32) && !defined(_WIN32_WCE) && !defined(__WIN32__) // Only defining the code on Windows
    return nullptr;
#else
    char volume_path[MAX_PATH];
    if (!GetVolumePathName(filePath, volume_path, ARRAYSIZE(volume_path)))
        return nullptr;

    char volume_name[MAX_PATH];
    if (!GetVolumeNameForVolumeMountPoint(volume_path,
        volume_name, ARRAYSIZE(volume_name)))
        return nullptr;

    auto length = strlen(volume_name);
    if (length && volume_name[length - 1] == L'\\')
        volume_name[length - 1] = L'\0';

    return CreateFile(volume_name, 0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
#endif
}

bool Utils::is_file_on_ssd(const char* file_path)
{
#if !defined(_WIN32) && !defined(_WIN32_WCE) && !defined(__WIN32__)
    // Not on Windows, haven't written the code to determine that on Linux yet
    return false;
#else
    bool is_ssd{ false };
    HANDLE volume = get_volume_handle_for_file(file_path);
    if (volume == INVALID_HANDLE_VALUE)
    {
        return false; /*invalid path! throw?*/
    }

    STORAGE_PROPERTY_QUERY query{};
    query.PropertyId = StorageDeviceSeekPenaltyProperty;
    query.QueryType = PropertyStandardQuery;
    DWORD count;
    DEVICE_SEEK_PENALTY_DESCRIPTOR result{};
    if (DeviceIoControl(volume, IOCTL_STORAGE_QUERY_PROPERTY,
        &query, sizeof(query), &result, sizeof(result), &count, nullptr))
    {
        is_ssd = !result.IncursSeekPenalty;
    }
    else { /*fails for network path, etc*/ }
    CloseHandle(volume);
    return is_ssd;
#endif
}

Image32Bit Utils::OIDN_denoise(const Image32Bit& image, int width, int height, float blend_factor)
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
                    g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "There was an error getting the device for denoising with OIDN. Perhaps some missing libraries for your hardware?");
                    return Image32Bit();
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
            g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "There was an error getting the device for denoising with OIDN. Perhaps some missing libraries for your hardware?");
            return Image32Bit();
        }
        device.commit();

        device_done = true;
    }

    if (!device_done)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Cannot create any OIDN device, aborting denoising...");
        return Image32Bit();
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

            colorPtr[index * 3 + 0] = image[index * 3 + 0];
            colorPtr[index * 3 + 1] = image[index * 3 + 1];
            colorPtr[index * 3 + 2] = image[index * 3 + 2];
        }
    // Filter the beauty image

    filter.execute();

    float* denoised_ptr = (float*)colorBuf.getData();
    Image32Bit output_image(width, height, 3);
    ColorRGB32F* output_pixels = output_image.get_data_as_ColorRGB32F();
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            ColorRGB32F color = blend_factor * ColorRGB32F(denoised_ptr[index * 3 + 0], denoised_ptr[index * 3 + 1], denoised_ptr[index * 3 + 2]) + (1.0f - blend_factor) * image.get_pixel_ColorRGB32F(index);

            output_pixels[index] = color;
        }

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Error: %s", errorMessage);

    return output_image;
}

void Utils::debugbreak()
{
#if defined( _WIN32 )
    __debugbreak();
#elif defined( __GNUC__ )
    // raise(SIGTRAP);
#else
    ;
#endif
}
