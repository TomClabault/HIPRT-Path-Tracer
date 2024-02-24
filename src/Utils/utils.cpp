//Already defined in image_io.cpp from gkit
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Image/color.h"
#include "Image/image.h"
#include "Image/image_io.h"
#include "Utils/utils.h"

#include <iostream>
#include <string>

#include <OpenImageDenoise/oidn.hpp>

std::vector<HIPRTColor> Utils::read_image_float(const std::string& filepath, int& image_width, int& image_height, bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int channels;
    float* pixels = stbi_loadf(filepath.c_str(), &image_width, &image_height, &channels, 0);

    if(!pixels)
    {
        std::cout << "Error reading image " << filepath << std::endl;
        std::exit(1);
    }

    std::vector<HIPRTColor> output(image_width * image_height);
    for (int y = 0; y < image_height; y++)
    {
        for (int x = 0; x < image_width; x++)
        {
            int index = y * image_width + x;
            output[index] = HIPRTColor(pixels[index * 3 + 0], pixels[index * 3 + 1], pixels[index * 3 + 2]);
        }
    }

    return output;
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<HIPRTColor>& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(reinterpret_cast<const float*>(hdr_image.data()), hdr_image.size(), sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure)
{
    return tonemap_hdr_image(hdr_image.data(), hdr_image.size(), sample_number, gamma, exposure);
}

std::vector<unsigned char> Utils::tonemap_hdr_image(const float* hdr_image, size_t size, int sample_number, float gamma, float exposure)
{
    std::vector<unsigned char> tonemapped_data(size);

#pragma omp parallel for
    for (int i = 0; i < size; i += 3)
    {
        Color pixel = Color(hdr_image[i + 0], hdr_image[i + 1], hdr_image[i + 2], 1.0f) / (float)sample_number;
        Color tone_mapped = Color(1.0f, 1.0f, 1.0f, 1.0f) - exp(-pixel * exposure);
        Color gamma_corrected = pow(tone_mapped, 1.0f / gamma);

        tonemapped_data[i + 0] = gamma_corrected.r * 255.0f;
        tonemapped_data[i + 1] = gamma_corrected.g * 255.0f;
        tonemapped_data[i + 2] = gamma_corrected.b * 255.0f;
    }

    return tonemapped_data;
}

std::vector<float> Utils::compute_env_map_cdf(const std::vector<HIPRTColor>& skysphere, int width, int height)
{
    std::vector<float> out(height * width);
    out[0] = 0.0f;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            out[index] = out[std::max(index - 1, 0)] + Utils::luminance_of_pixel(skysphere, width, x, y);
        }
    }

    return out;
}

std::vector< HIPRTColor> Utils::OIDN_denoise(const std::vector<HIPRTColor>& image, int width, int height, float blend_factor)
{
    // Create an Open Image Denoise device
    static bool device_done = false;
    static oidn::DeviceRef device;
    if (!device_done)
    {
        device = oidn::newDevice(); // CPU or GPU if available
        if (device == NULL)
        {
            std::cerr << "There was an error getting the device for denoising with OIDN. Perhaps some missing DLLs for your hardware?" << std::endl;
            return std::vector<HIPRTColor>();
        }
        device.commit();

        device_done = true;
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
    std::vector<HIPRTColor> output(width * height);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            HIPRTColor color = blend_factor * HIPRTColor(denoised_ptr[index * 3 + 0], denoised_ptr[index * 3 + 1], denoised_ptr[index * 3 + 2])
                + (1.0f - blend_factor) * image[index];
            color.a = 1.0f;

            output[index] = color;
        }

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cout << "Error: " << errorMessage << std::endl;

    return output;
}
std::vector<ImageBin> Utils::importance_split_skysphere(const std::vector<HIPRTColor>& skysphere, int width, int height, ImageBin current_region, float current_radiance, int minimum_bin_area, float minimum_bin_radiance)
{
    int horizontal_extent = current_region.x1 - current_region.x0;
    int vertical_extent = current_region.y1 - current_region.y0;
    int current_region_area = vertical_extent * vertical_extent;
    if (current_radiance <= minimum_bin_radiance || current_region_area <= minimum_bin_area)
        return std::vector<ImageBin> { current_region };

    //Determining the largest for the split
    //A vertical split means that the "cut line" is horizontal, 
    //we're dividing the height by 2
    bool vertical_split = horizontal_extent < vertical_extent;

    ImageBin new_region_1;
    ImageBin new_region_2;
    if (vertical_split)
    {
        new_region_1 = ImageBin { current_region.x0, current_region.x1, 
                                  current_region.y0, vertical_extent / 2 + current_region.y0};
        new_region_2 = ImageBin { current_region.x0, current_region.x1, 
                                  vertical_extent / 2 + current_region.y0, current_region.y1 };
    }
    else
    {
        new_region_1 = ImageBin{ current_region.x0, horizontal_extent / 2 + current_region.x0,
                                 current_region.y0, current_region.y1 };
        new_region_2 = ImageBin{ horizontal_extent / 2 + current_region.x0, current_region.x1,
                                 current_region.y0, current_region.y1 };
    }

    float region_1_radiance = Utils::luminance_of_area(skysphere, width, height, new_region_1);
    float region_2_radiance = Utils::luminance_of_area(skysphere, width, height, new_region_2);

    std::vector<ImageBin> region_1_bins = Utils::importance_split_skysphere(skysphere, width, height, new_region_1, region_1_radiance, minimum_bin_area, minimum_bin_radiance);
    std::vector<ImageBin> region_2_bins = Utils::importance_split_skysphere(skysphere, width, height, new_region_2, region_2_radiance, minimum_bin_area, minimum_bin_radiance);

    std::vector<ImageBin> all_bins;
    all_bins.insert(all_bins.end(), region_1_bins.begin(), region_1_bins.end());
    all_bins.insert(all_bins.end(), region_2_bins.begin(), region_2_bins.end());

    return all_bins;
}

float Utils::luminance_of_pixel(const std::vector<HIPRTColor>& skysphere, int width, int x, int y)
{
    HIPRTColor pixel = skysphere[y * width + x];

    return 0.3086 * pixel.r + 0.6094 * pixel.g + 0.0820 * pixel.b;
}

float Utils::luminance_of_area(const std::vector<HIPRTColor>& skysphere, int width, int start_x, int start_y, int stop_x, int stop_y)
{
    float luminance = 0.0f;

    for (int x = start_x; x < stop_x; x++)
        for (int y = start_y; y < stop_y; y++)
            luminance += Utils::luminance_of_pixel(skysphere, width, x, y);

    return luminance;
}

float Utils::luminance_of_area(const std::vector<HIPRTColor>& skysphere, int width, int height, const ImageBin& area)
{
    return Utils::luminance_of_area(skysphere, width, area.x0, area.y0, area.x1, area.y1);
}

std::vector<ImageBin> Utils::importance_split_skysphere(const std::vector<HIPRTColor>& skysphere, int width, int height, int minimum_bin_area, float minimum_bin_radiance)
{
    ImageBin whole_image_region = ImageBin{ 0, width, 0, height };

    float current_radiance = Utils::luminance_of_area(skysphere, width, height, whole_image_region);

    return Utils::importance_split_skysphere(skysphere, width, height, whole_image_region, current_radiance, minimum_bin_area, minimum_bin_radiance);
}

void Utils::write_env_map_bins_to_file(const std::string& filepath, std::vector<HIPRTColor> skysphere_data, int width, int height, const std::vector<ImageBin>& skysphere_importance_bins)
{
    int max_index = width * height - 1;

    for (const ImageBin& bin : skysphere_importance_bins)
    {
        for (int y = bin.y0; y < bin.y1; y++)
        {
            int index1 = std::min(y * width + bin.x0, max_index);
            int index2 = std::min(y * width + bin.x1, max_index);

            skysphere_data[index1] = HIPRTColor(1.0f, 0.0f, 0.0f);
            skysphere_data[index2] = HIPRTColor(1.0f, 0.0f, 0.0f);
        }

        for (int x = bin.x0; x < bin.x1; x++)
        {
            int index1 = std::min(bin.y0 * width + x, max_index);
            int index2 = std::min(bin.y1 * width + x, max_index);

            skysphere_data[index1] = HIPRTColor(1.0f, 0.0f, 0.0f);
            skysphere_data[index2] = HIPRTColor(1.0f, 0.0f, 0.0f);
        }
    }

    write_image_hdr(skysphere_data, width, height, filepath.c_str());
}
