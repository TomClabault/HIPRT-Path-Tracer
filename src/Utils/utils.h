/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UTILS_H
#define UTILS_H

#include "Image/image.h"
#include "HostDeviceCommon/color_rgb.h"

#include <string>

/*
 * Defines a rectangular region on a skysphere / image
 */

class Utils
{
public:

    static Image read_image_float(const std::string& filepath, int& image_width, int& image_height, bool flipY = true);
    static std::vector<unsigned char> tonemap_hdr_image(const Image& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<ColorRGB>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const std::vector<float>& hdr_image, int sample_number, float gamma, float exposure);
    static std::vector<unsigned char> tonemap_hdr_image(const float* hdr_image, size_t size, int sample_number, float gamma, float exposure);

    static std::string file_to_string(const char* filepath);

    /*
     * A blend factor of 1 gives only the noisy image. 0 only the denoised image
     */
    static Image OIDN_denoise(const Image& image, int width, int height, float blend_factor);
};

#endif
