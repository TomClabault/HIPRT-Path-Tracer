/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "stb_image.h"

#include "Image/Image.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"
#include "FLIP.h"
#include "clip.h"

#include <deque>
#include <format>
#include <iostream>
#include <iomanip> // get_current_date_string()
#include <OpenImageDenoise/oidn.hpp>
#include <sstream>
#include <string>
#include <string_view>

#if defined(_WIN32) || defined(_WIN32_WCE) || defined(__WIN32__)
#include <Windows.h> // for is_file_on_SSD() and other functions
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

void Utils::compute_alias_table(const std::vector<float>& input, float in_input_total_sum, std::vector<float>& out_probas, std::vector<int>& out_alias)
{
    if (input.size() == 0)
        return;

    // A vector of the luminance of all the pixels of the envmap
    // normalized such that the average of the elements of this vector is 'width*height'
    double input_total_sum_double = in_input_total_sum;
    std::vector<double> normalized_elements(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        // Normalized
        normalized_elements.at(i) = static_cast<double>(input.at(i)) / input_total_sum_double;

        // Scale for alias table construction such that the average of
        // the elements is 1
        normalized_elements.at(i) *= input.size();
    }

    out_probas.resize(input.size());
    out_alias.resize(input.size());

    std::deque<int> smalls;
    std::deque<int> larges;

    for (int i = 0; i < normalized_elements.size(); i++)
    {
        if (normalized_elements.at(i) < 1.0f)
            smalls.push_back(i);
        else
            larges.push_back(i);
    }

    while (!smalls.empty() && !larges.empty())
    {
        int small_index = smalls.front();
        int large_index = larges.front();

        smalls.pop_front();
        larges.pop_front();

        out_probas.at(small_index) = normalized_elements.at(small_index);
        out_alias.at(small_index) = large_index;

        normalized_elements.at(large_index) = (normalized_elements.at(large_index) + normalized_elements.at(small_index)) - 1.0f;
        if (normalized_elements.at(large_index) > 1.0f)
            larges.push_back(large_index);
        else
            smalls.push_back(large_index);
    }

    while (!larges.empty())
    {
        int index = larges.front();
        larges.pop_front();

        out_probas.at(index) = 1.0f;
    }

    while (!smalls.empty())
    {
        int index = smalls.front();
        smalls.pop_front();

        out_probas.at(index) = 1.0f;
    }
}

void Utils::compute_alias_table(const std::vector<float>& input, std::vector<float>& out_probas, std::vector<int>& out_alias, float* out_input_total_sum)
{
    float sum = 0.0f;
    for (float input_element : input)
        sum += input_element;

    if (out_input_total_sum)
        *out_input_total_sum = sum;

    Utils::compute_alias_table(input, sum, out_probas, out_alias);
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

#include "tinyfiledialogs.h"

std::string Utils::open_file_dialog(const char* filter_patterns[], int filter_count)
{
    const char* file = tinyfd_openFileDialog("let us read the password back", "", filter_count, filter_patterns, NULL, 0);
    if (file)
        return std::string(file);
    else
        return "";
}

float Utils::compute_image_mse(const Image32Bit& reference, const Image32Bit& subject)
{
    float mse = 0.0f;

    if (reference.width != subject.width || reference.height != subject.height)
    {
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Cannot compute difference between images of different sizes.");
        return mse;
    }

    for (int i = 0; i < reference.width * reference.height; i++)
    {
        ColorRGB32F reference_pixel = reference.get_pixel_ColorRGB32F(i);
        ColorRGB32F subject_pixel = subject.get_pixel_ColorRGB32F(i);

        float diff_r_2 = hippt::square(reference_pixel.r - subject_pixel.r);
        float diff_g_2 = hippt::square(reference_pixel.g - subject_pixel.g);
        float diff_b_2 = hippt::square(reference_pixel.b - subject_pixel.b);

        mse += diff_r_2 + diff_g_2 + diff_b_2;
    }

    mse /= static_cast<float>(reference.width * reference.height);

    return mse;
}

float Utils::compute_image_rmse(const Image32Bit& reference, const Image32Bit& subject)
{
    return sqrtf(Utils::compute_image_mse(reference, subject));
}

float Utils::compute_image_weighted_median_FLIP(const Image32Bit& reference_srgb, const Image32Bit& subject_srgb, float** out_error_map)
{
	float mean_flip_error = 0.0f;

    Image32Bit reference = reference_srgb.to_linear_rgb();
    Image32Bit subject = subject_srgb.to_linear_rgb();

    FLIP::Parameters parameters;
    FLIP::evaluate(reference.data().data(), subject.data().data(), reference.width, reference.height, false, parameters, true, true, mean_flip_error, out_error_map);

    return mean_flip_error;
}

void Utils::copy_u8_image_data_to_clipboard(const std::vector<unsigned char>& data, int width, int height)
{
    clip::image_spec spec;
    spec.width = width;
    spec.height = height;
    spec.bits_per_pixel = 32;
    spec.bytes_per_row = spec.width * 4;
    spec.red_mask = 0xff;
    spec.green_mask = 0xff00;
    spec.blue_mask = 0xff0000;
    spec.alpha_mask = 0xff000000;
    spec.red_shift = 0;
    spec.green_shift = 8;
    spec.blue_shift = 16;
    spec.alpha_shift = 24;
    clip::image img(data.data(), spec);

    if (!clip::set_image(img))
        g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Failed to copy image to clipboard.");
}

void Utils::copy_image_to_clipboard(const Image8Bit& image)
{
    std::vector<unsigned char> flipped_data(image.width * image.height * 4);
    for (int y = 0; y < image.height; y++)
    {
        for (int x = 0; x < image.width; x++)
        {
            int input_index = (x + y * image.width) * image.channels;
            int output_index = (x + (image.height - 1 - y) * image.width) * 4;

            flipped_data[output_index + 0] = image.data().data()[input_index + 0];
            flipped_data[output_index + 1] = image.data().data()[input_index + 1];
            flipped_data[output_index + 2] = image.data().data()[input_index + 2];
            flipped_data[output_index + 3] = 255;
        }
    }

    copy_u8_image_data_to_clipboard(flipped_data, image.width, image.height);
}

void Utils::copy_image_to_clipboard(const Image32Bit& image)
{
    std::vector<unsigned char> image_data_8u(image.width * image.height * 4);
    for (int y = 0; y < image.height; y++)
    {
        for (int x = 0; x < image.width; x++)
        {
            int input_index = (x + y * image.width) * image.channels;
            int output_index = (x + (image.height - 1 - y) * image.width) * 4;

            image_data_8u[output_index + 0] = static_cast<unsigned char>(hippt::clamp(0.0f, 1.0f, image.data().data()[input_index + 0]) * 255.0f);
            image_data_8u[output_index + 1] = static_cast<unsigned char>(hippt::clamp(0.0f, 1.0f, image.data().data()[input_index + 1]) * 255.0f);
            image_data_8u[output_index + 2] = static_cast<unsigned char>(hippt::clamp(0.0f, 1.0f, image.data().data()[input_index + 2]) * 255.0f);
            image_data_8u[output_index + 3] = 255;
        }
    }

    copy_u8_image_data_to_clipboard(image_data_8u, image.width, image.height);
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
    raise(SIGTRAP);
#else
    ;
#endif
}

#ifdef _WIN32
// Code from @jpownby from the GraphicsProgramming Discord
Utils::AddEnvVarError Utils::windows_add_ENV_var_to_PATH(const wchar_t* env_var_name, std::wstring extra_string)
{
    // Get $CUDA_PATH
    // (this code assumes that the path isn't longer than MAX_PATH;
    // if you want to be robust against a machine configured for long paths
    // you could use instead use two calls and dynamically allocate the string
    // as shown below for $PATH)
    wchar_t envVarBuffer[MAX_PATH] = { L'\0' };
    DWORD envVarValueLength_notIncludingNull = 0;
    {
        const auto result = GetEnvironmentVariableW(env_var_name, envVarBuffer, MAX_PATH);
        if (result != 0)
        {
            if (result < MAX_PATH)
                envVarValueLength_notIncludingNull = result;
            else
                // $CUDA_PATH is longer than MAX_PATH; the machine must be configured for long paths
                return AddEnvVarError::ADD_ENV_VAR_ERROR_VALUE_TOO_LONG;
        }
        else
        {
            const auto errorCode = GetLastError();
            if (errorCode == ERROR_ENVVAR_NOT_FOUND)
                // The given environment variable doesn't exist
                return AddEnvVarError::ADD_ENV_VAR_ERROR_NOT_FOUND;
            else
                return AddEnvVarError::ADD_ENV_VAR_ERROR_UNKNOWN;
        }
    }

    if (envVarValueLength_notIncludingNull > 0)
    {
        // You could statically allocate an array and hope that it's big enough,
        // but this code instead makes two calls to GetEnvironmentVariableW() and dynamically allocates the exact amount

        // Get the length of the current $PATH
        constexpr auto* const environmentVariableName = L"PATH";
        DWORD codeUnitCountOfExistingPath_includingTerminatingNull = 0;
        {
            constexpr DWORD returnRequiredSize = 0;
            codeUnitCountOfExistingPath_includingTerminatingNull = GetEnvironmentVariableW(environmentVariableName, nullptr, returnRequiredSize);
            if (codeUnitCountOfExistingPath_includingTerminatingNull == 0)
            {
                const auto errorCode = GetLastError();
                if (errorCode == ERROR_ENVVAR_NOT_FOUND)
                    // $PATH doesn't exist
                    codeUnitCountOfExistingPath_includingTerminatingNull = 1; // 0 + NULL
                else
                    return AddEnvVarError::ADD_ENV_VAR_ERROR_UNKNOWN;
            }
        }
        // Allocate enough space for the current $PATH and the extra path to add
        const auto pathToAdd = std::format(L";{}{}", std::wstring_view(envVarBuffer, envVarValueLength_notIncludingNull), extra_string);
        const auto codeUnitCountRequired_includingTerminatingNull = codeUnitCountOfExistingPath_includingTerminatingNull + pathToAdd.length();
        std::wstring path((codeUnitCountRequired_includingTerminatingNull - 1), L'\0');   // std::wstring automatically deals with the terminating NULL
        // Get the current $PATH
        {
            const auto result = GetEnvironmentVariableW(environmentVariableName, path.data(), codeUnitCountRequired_includingTerminatingNull);
            if (result != 0)
            {
                if (result <= path.length())
                {
                    const auto codeUnitCountOfExistingPath_notIncludingTerminatingNull = result;
                    path.resize(codeUnitCountOfExistingPath_notIncludingTerminatingNull);
                }
                else
                {
                    // Another process/thread must have changed $PATH to be larger? :/
                    // const auto codeUnitCountOfExistingPath_includingTerminatingNull = result;

                    return AddEnvVarError::ADD_ENV_VAR_ERROR_UNKNOWN;
                }
            }
            else
            {
                // An error happened :(
                // const auto errorCode = GetLastError();

                return AddEnvVarError::ADD_ENV_VAR_ERROR_UNKNOWN;
            }
        }

        // Append the new path
        path.append(pathToAdd);
        // Set the updated $PATH
        if (!SetEnvironmentVariableW(environmentVariableName, path.c_str()))
        {
            // An error happened :(
            // const auto errorCode = GetLastError();

            return AddEnvVarError::ADD_ENV_VAR_ERROR_UNKNOWN;
        }
    }

    return AddEnvVarError::ADD_ENV_VAR_ERROR_NONE;
}
#endif
